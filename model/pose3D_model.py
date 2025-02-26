import time

import torch
import torch.nn as nn
from model.swin_space import SwinTransformerspace
from model.swin_time import PoseTransformer


from utils.loss import mpjpe

from utils.projecter import project


class RefinedFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinedFusion, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm([4, out_channels, 768])  # Adjust dimensions if necessary

    def forward(self, x):
        identity = x
        out = self.bn(x)
        out = self.conv(out)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        # x = self.reduction(x)

        return x


class Attention2(nn.Module):
    def __init__(
            self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        V1 = x1
        V2 = x2

        B, N, C = V1.shape
        qkv = self.qkv(V1)

        V2 = V2.reshape(N, 1, self.num_heads, -1).permute(1, 2, 0, 3)
        qkv = qkv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k = qkv[0], qkv[1]  # make torchscript happy (cannot use tensor as tuple)
        v = V2

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Pose3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.space = SwinTransformerspace()

        self.time = PoseTransformer()

        self.input_resolution = [224, 224]
        self.dim = 3

        self.attn = Attention2(768)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool10 = nn.AdaptiveAvgPool1d(10)
        self.embed_dim = 768

        # self.regression = nn.Linear(self.embed_dim, 51)

        self.layers = nn.ModuleList()
        for i_layer in range(5):
            layer = PatchMerging(
                (self.input_resolution[0] // (2 ** i_layer), self.input_resolution[1] // (2 ** i_layer)),
                dim=int(self.dim * 4 ** i_layer))
            self.layers.append(layer)


        self.rayconv = nn.Linear(self.embed_dim + 3072, self.embed_dim)
        self.refined = RefinedFusion(in_channels=8, out_channels=8)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, 8, self.embed_dim))

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 51),
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.mlp = Mlp(in_features=self.embed_dim, hidden_features=int(self.embed_dim * 4), act_layer=nn.GELU, drop=0.)

    def forward_features2(self, x):
        x = self.avgpool(x.transpose(1, 2))  # B C 1    torch.Size([16, 768, 1])
        x = x.permute(2, 0, 1)

        return x

    def forward(self, x, rays_d, targets, loss_total):
        B, H, W, C = rays_d.shape
        rays_d = rays_d.view(B, -1, C)

        for layer in self.layers:
            rays_d = layer(rays_d)


        x, loss_total, loss, coords_spatial = self.space(x, targets, loss_total)
        coords_spatial = coords_spatial.view(-1, 17, 3)

        loss_2d_all = 0
        for i in range(coords_spatial.view(-1, 17, 3).size(0)):
            loss_2d = project(coords_spatial.view(-1, 17, 3)[i], targets[i], i % 4)
            loss_2d_all = loss_2d_all + loss_2d

        loss_2d_all = loss_2d_all / coords_spatial.view(-1, 17, 3).size(0)

        loss_total = loss_total + loss_2d_all
        # print(x.shape)
        space_feature = torch.cat((x, rays_d), dim=-1)
        space_feature = self.rayconv(space_feature)
        space_feature = self.norm(space_feature)
        #
        space_feature = self.avgpool(space_feature.transpose(1, 2))  # B C 1    torch.Size([16, 768, 1])
        space_feature = space_feature.permute(2, 0, 1)
        #
        # print(space_feature.shape)
        #
        new_targets = [targets[i::4] for i in range(4)]
        views = [space_feature[i::4] for i in range(4)]  # 视角列表
        #
        all_views = []
        #
        for i in range(4):
            # permute 和池化操作
            views[i] = self.part(views[i].view(1, 8, -1))
        #
        for i in range(4):
            all_view, loss_total, _ = self.time(views[i], new_targets[i], loss_total)
            all_views.append(torch.squeeze(all_view))
        #
        result = torch.cat(all_views, dim=0)

        coords_spatial_s = [coords_spatial[i::4] for i in range(4)]

        losses = [0] * 4
        loss_2d_all = 0
        # # 计算每个视角的投影误差

        for i in range(8):
            for j in range(4):
                losses[j] += project(coords_spatial_s[j][i], new_targets[i], j)
                loss_2d_all = loss_2d_all + project(coords_spatial_s[j][i], new_targets[i], j)

        loss_view = 0
        for j in range(4):
            losses[j] = losses[j] / 8 + mpjpe(coords_spatial_s[j], new_targets)
            loss_view = loss_view + mpjpe(coords_spatial_s[j], new_targets)
        #
        loss_total = loss_total + loss_2d_all + loss_view
        #
        inverse_losses = [1 / loss if loss > 0 else 1e-8 for loss in losses]
        total_inverse_loss = sum(inverse_losses)
        weights = [il / total_inverse_loss for il in inverse_losses]  # 归一化权重
        #
        fused_feature = torch.zeros_like(views[0])  # 和每个视角特征相同的形状
        #
        # # 使用权重进行特征融合
        for j in range(4):
            fused_feature += weights[j] * views[j]
        fused_feature = self.avgpool(fused_feature.transpose(1, 2)).permute(2, 0, 1)  # B C 1    torch.Size([16, 768, 1])

        all_view, loss_total = self.time(fused_feature, new_targets, loss_total)

        coords = self.head(all_view)
        return coords.view(-1, 17, 3), loss_total

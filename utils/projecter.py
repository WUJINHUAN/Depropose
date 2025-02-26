import numpy as np
import json
from utils.loss import mpjpe
import torch
import time


camera_all=[{"R": [[-0.9153617321513369, 0.40180836633680234, 0.02574754463350265], [0.051548117060134555, 0.1803735689384521, -0.9822464900705729], [-0.399319034032262, -0.8977836111057917, -0.185819527201491]], "T": [[1841.10702774543], [4955.28462344526], [1563.4453958977]], "fx": [1145.04940458804], "fy": [1143.78109572365], "cx": [512.541504956548], "cy": [515.4514869776], "k": [[-0.207098910824901], [0.247775183068982], [-0.00307515035078854]], "p": [[-0.00142447157470321], [-0.000975698859470499]], "K": [[1145.04940458804, 0.0, 512.541504956548], [0.0, 1143.78109572365, 515.4514869776], [0.0, 0.0, 1.0]]}
    ,{"R": [[0.9281683400814921, 0.3721538354721445, 0.002248380248018696], [0.08166409428175585, -0.1977722953267526, -0.976840363061605], [-0.3630902204349604, 0.9068559102440475, -0.21395758897485287]], "T": [[1761.27853428116], [-5078.00659454077], [1606.2649598335]], "fx": [1149.67569986785], "fy": [1147.59161666764], "cx": [508.848621645943], "cy": [508.064917088557], "k": [[-0.194213629607385], [0.240408539138292], [0.00681997559022603]], "p": [[-0.0027408943961907], [-0.001619026613787]], "K": [[1149.67569986785, 0.0, 508.848621645943], [0.0, 1147.59161666764, 508.064917088557], [0.0, 0.0, 1.0]]}
    ,{"R": [[-0.9141549520542256, -0.40277802228118775, -0.045722952682337906], [-0.04562341383935874, 0.21430849526487267, -0.9756999400261069], [0.4027893093720077, -0.889854894701693, -0.214287280609606]], "T": [[-1846.7776610084], [5215.04650469073], [1491.97246576518]], "fx": [1149.14071676148], "fy": [1148.7989685676], "cx": [519.815837182153], "cy": [501.402658888552], "k": [[-0.208338188251856], [0.255488007488945], [-0.00246049749891915]], "p": [[-0.000759999321030303], [0.00148438698385668]], "K": [[1149.14071676148, 0.0, 519.815837182153], [0.0, 1148.7989685676, 501.402658888552], [0.0, 0.0, 1.0]]}
    ,{"R": [[0.9141562410494211, -0.40060705854636447, 0.061905989962380774], [-0.05641000739510571, -0.2769531972942539, -0.9592261660183036], [0.40141783470104664, 0.8733904688919611, -0.2757767409202658]], "T": [[-1794.78972871109], [-3722.69891503676], [1574.89272604599]], "fx": [1145.51133842318], "fy": [1144.77392807652], "cx": [514.968197319863], "cy": [501.882018537695], "k": [[-0.198384093827848], [0.218323676298049], [-0.00894780704152122]], "p": [[-0.00181336200488089], [-0.000587205583421232]], "K": [[1145.51133842318, 0.0, 514.968197319863], [0.0, 1144.77392807652, 501.882018537695], [0.0, 0.0, 1.0]]}
    ]

camera_params = []
for i in range(4):
    camera = camera_all[i]  # 读取相机参数
    camera_tensor = {
        "R": torch.tensor(camera["R"], dtype=torch.float32).to('cuda'),
        "T": torch.tensor(camera["T"], dtype=torch.float32).to('cuda'),
        "f": torch.tensor([camera["fx"], camera["fy"]], dtype=torch.float32).to('cuda'),
        "c": torch.tensor([camera["cx"], camera["cy"]], dtype=torch.float32).to('cuda'),
        "k": torch.tensor(camera["k"], dtype=torch.float32).to('cuda'),
        "p": torch.tensor(camera["p"], dtype=torch.float32).to('cuda'),
    }
    camera_params.append(camera_tensor)

def project_point_radial(P, i):
    """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """
    # P is a matrix of 3-dimensional points


    f, c, R, T, k, p = (
    camera_params[i]["f"], camera_params[i]["c"], camera_params[i]["R"], camera_params[i]["T"], camera_params[i]["k"],
    camera_params[i]["p"])

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    #X = R.dot(P.T - T)  # rotate and translate

    P_centered = P - T.T
    X = torch.matmul(R, P_centered.T)
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    #radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    #radial = torch.zeros(N, dtype=torch.float32).to('cuda')
    # for i in range(N):
    #     # 对于第 i 个点，计算径向畸变
    #     radial[i] = 1 + k[0] * r2[i] + k[1] * (r2[i] ** 2) + k[2] * (r2[i] ** 3)
    radial = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * (radial + tan) + p.reshape(2, 1) * r2

    Proj = (f * XXX) + c
    Proj = Proj.permute(1, 0)

    D = X[2,]

    return Proj, D, radial, tan, r2

def project(pre_joint, target, i):
    pre_2d, _, _, _, _ = project_point_radial(pre_joint, i)
    tra_2d, _, _, _, _ = project_point_radial(target, i)
    loss = mpjpe(pre_2d, tra_2d)

    return loss

#
# annotation_path = r"D:\Pythoncoda\new_model\test\action1_2.json"
# with open(annotation_path, 'r') as f:
#     dataset = json.load(f)
#
# joints_3d = dataset["images"]["s_01_Directions 1"]["ca_01"][0]["joints_3d"]
# joints_2d = dataset["images"]["s_01_Directions 1"]["ca_01"][0]["joints_2d"]
#
# start_time = time.time()
# for i in range(100):
# # print(pre_2d)
# # print(joints_2d)
#     loss = project(torch.tensor(joints_3d).to("cuda"),torch.tensor(joints_3d).to("cuda"),0)
# end_time = time.time()
# print(f"运行时间: {end_time - start_time:.4f} 秒")
# print(loss)

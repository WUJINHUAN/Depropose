import re
import os


def extract_action(filename):
    # 使用正则表达式提取以 "Directions" 为关键字的部分
    match = re.search(r"s_\d+_(\w+)", filename)
    if match:
        return match.group(1)  # 返回匹配到的 "Directions"
    return None  # 如果没有找到，返回 None


# # 示例调用
# input_directory = r'E:\noise\pythonProject\S9'
#
# # 获取输入目录下的所有子文件夹
# subdirectories = [f.path for f in os.scandir(input_directory) if f.is_dir()]
# extracted_names=[]
# # 遍历每个子文件夹，并将其中的内容移动到目标目录
# for subdirectory in subdirectories:
#     name = extract_action(subdirectory)
#     extracted_names.append(name)
#
# if len(set(extracted_names)) == 1:  # 如果 set 中只有一个元素，说明所有元素都相同
#         print("yes")
# else:
#         print("no")
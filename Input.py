"""
2024.03.21
author:alian
数据预处理操作
1.数据集分割
"""
import os
import random
import shutil
import numpy as np


def txt_to_bin(input_txt_path, output_dir):
    """
    将3D点云数据从TXT格式转换为BIN格式

    参数:
        input_txt_path (str): 输入TXT文件路径
        output_dir (str): 输出目录路径

    返回:
        str: 生成的BIN文件路径
    """

    # 读取TXT文件
    with open(input_txt_path, 'r') as f:
        lines = f.readlines()

    # 解析数据 - 每行包含4个浮点数(x,y,z,intensity)
    points = []
    for line in lines:
        # 跳过空行
        if not line.strip():
            continue

        # 分割每行的数据
        parts = line.strip().split()
        if len(parts) != 4:
            continue

        # 转换为浮点数
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            intensity = float(parts[3])
            points.append([x, y, z, intensity])
        except ValueError:
            continue

    # 转换为numpy数组
    point_cloud = np.array(points, dtype=np.float32)

    # 构建输出路径 - 保持文件名不变，只修改扩展名
    filename = os.path.basename(input_txt_path)
    filename_without_ext = os.path.splitext(filename)[0]
    output_bin_path = os.path.join(output_dir)

    # 写入BIN文件
    point_cloud.tofile(output_bin_path)

    #return output_bin_path


# 使用示例
# input_path = "path/to/your/pointcloud.txt"
# output_path = "path/to/output/directory"
# bin_file = txt_to_bin(input_path, output_path)
# print(f"BIN文件已保存到: {bin_file}")

def rename_and_copy_os(src_dir, dst_dir, start_num=7518):
    """使用os模块的实现"""
    # 清空目标文件夹
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    # 获取并排序文件
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.txt'))

    for i, filename in enumerate(files, start=start_num):
        new_name = f"{i:06d}.bin" #################### bin
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)
        txt_to_bin(src_path, dst_path)
        #shutil.copy2(src_path, dst_path)

def copy_os(src_dir, calib_txt, calib_folder, start_num=7518):
    """使用os模块的实现"""
    # 清空目标文件夹
    if os.path.exists(calib_folder):
        shutil.rmtree(calib_folder)
    os.makedirs(calib_folder)

    # 获取并排序文件
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.bin'))####################### bin

    for i, filename in enumerate(files, start=start_num):
        new_name = f"{i:06d}.txt"
        dst_path = os.path.join(calib_folder, new_name)
        shutil.copy2(calib_txt, dst_path)


if __name__=='__main__':
    """
        参考config.ini文件
        src_path 为 pointcloud的上一层
    """
    src_path = 'G:/3D/labelCloud/'  # Inpt数据目录
    pointcloud_folder = src_path+'pointclouds/'  # 导出数据目录
    calib_folder = src_path+ 'calib/'  # 导出校准文件目录
    Input_path = src_path + 'Input/'  # Inpt数据目录
    calib_txt = src_path+ 'calib.txt'

    rename_and_copy_os(Input_path, pointcloud_folder,7518)
    copy_os(pointcloud_folder,calib_txt,calib_folder,7518)






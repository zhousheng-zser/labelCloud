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
import struct
import glob

def read_pcd_points(pcd_file):
    points = []

    with open(pcd_file, 'r') as f:
        lines = f.readlines()

    # 跳过头部，找到 DATA ascii 后的行
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            data_start = i + 1
            break

    # 读取点
    for line in lines[data_start:]:
        if not line.strip():
            continue

        vals = line.strip().split(',')
        if len(vals) != 4:
            continue

        x, y, z, intensity = map(float, vals)
        points.append((x, y, z, intensity))
    return points

def save_kitti_bin(points, out_file):
    with open(out_file, 'wb') as f:
        for p in points:
            f.write(struct.pack('ffff', float(p[0]), float(p[1]), float(p[2]), float(p[3])))

def pcd_to_bin(input_txt_path, output_dir):
    """
    将3D点云数据从PCD格式转换为BIN格式

    参数:
        input_txt_path (str): 输入PCD文件路径
        output_dir (str): 输出目录路径

    返回:
        str: 生成的BIN文件路径
    """
    points = read_pcd_points(input_txt_path)
    save_kitti_bin(points, output_dir)


# 使用示例
# input_path = "path/to/your/pointcloud.txt"
# output_path = "path/to/output/directory"
# bin_file = pcd_to_bin(input_path, output_path)
# print(f"BIN文件已保存到: {bin_file}")

def rename_and_copy_os(src_dir, dst_dir, start_num=8586):
    """使用os模块的实现"""
    # 清空目标文件夹
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    # 获取并排序文件
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.pcd'))

    for i, filename in enumerate(files, start=start_num):
        new_name = f"{i:06d}.bin" #################### bin
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)
        pcd_to_bin(src_path, dst_path)
        #shutil.copy2(src_path, dst_path)

def copy_os(src_dir, calib_txt, calib_folder, start_num=8586):
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
    id = int(input("请输入开始下标："))
    rename_and_copy_os(Input_path, pointcloud_folder,id)
    copy_os(pointcloud_folder,calib_txt,calib_folder,id)






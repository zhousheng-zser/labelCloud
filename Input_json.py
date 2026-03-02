"""
2025.12.09
author:alian
数据预处理操作
1.数据集分割
"""
import os
import random
import shutil
import numpy as np
import struct
import json
import open3d as o3d
import glob
import math
from pathlib import Path

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

def hesai_to_kitti_intensity(raw_i):
    # 限制范围
    if raw_i > 255:
        raw_i = 255.0
    
    # KITTI 风格：log 非线性压缩（拟合 Velodyne 旧驱动）
    kitti_i = math.log1p(raw_i) / math.log1p(255.0)
    
    # 限幅到 0~1
    if kitti_i < 0:
        kitti_i = 0
    if kitti_i > 1:
        kitti_i = 1
    
    return kitti_i

def json_to_bin(input_txt_path, output_dir):
   
    #points = read_pcd_points(input_txt_path)
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        points = json.load(f)
    filtered_points = [pt for pt in points if len(pt) >= 4 and pt[3] >= 0.0]
    for i, pt in enumerate(filtered_points):
        x = pt[0] if len(pt) > 0 else 0.0
        y = pt[1] if len(pt) > 1 else 0.0
        z = pt[2] if len(pt) > 2 else 0.0
        raw_intensity = pt[3] if len(pt) > 3 else 0.0
            
        # 对于强度大于等于0.0的点，将强度值转换为KITTI风格
        intensity = hesai_to_kitti_intensity(raw_intensity)
        filtered_points[i] =[x,y,z,intensity]

    save_kitti_bin(filtered_points, output_dir)

def json_to_planse(input_txt_path, output_dir):
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        points = json.load(f)
    filtered_points = [[pt[0],pt[1],pt[2]] for pt in points ]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.4,  # 内点距离阈值（单位：米）
        ransac_n=3,  # 每次采样3个点
        num_iterations=10000  # 迭代次数
    )
    a, b, c, d = plane_model
    if b > 0:
        plane_model =  -plane_model
        a, b, c, d = plane_model

    with open(output_dir, 'w') as f:
        f.write('# Matrix\n')
        f.write('WIDTH 4\n')
        f.write('HEIGHT 1\n')
        f.write(f"{a:.6e} {b:.6e} {c:.6e} {d:.6e}\n")


# 使用示例
# input_path = "path/to/your/pointcloud.txt"
# output_path = "path/to/output/directory"
# bin_file = pcd_to_bin(input_path, output_path)
# print(f"BIN文件已保存到: {bin_file}")

def rename_and_copy_os(src_dir,planes_dir, dst_dir, start_num=8586):
    """使用os模块的实现"""
    # 清空目标文件夹
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    if os.path.exists(planes_dir):
        shutil.rmtree(planes_dir)
    os.makedirs(planes_dir)

    # 获取并排序文件
    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.json'))

    
    for i, filename in enumerate(files, start=start_num):
        new_name = f"{i:06d}.txt" #################### bin
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(planes_dir, new_name)
        json_to_planse(src_path, dst_path)

    for i, filename in enumerate(files, start=start_num):
        new_name = f"{i:06d}.bin" #################### bin
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, new_name)
        json_to_bin(src_path, dst_path)

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
    planes_path = src_path + 'planes/'  # 地面数据
    calib_txt = src_path+ 'calib.txt'
    id = int(input("请输入开始下标："))
    rename_and_copy_os(Input_path,planes_path, pointcloud_folder, id)
    copy_os(pointcloud_folder,calib_txt,calib_folder,id)






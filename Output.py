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
from draw_kitti_util import *
import open3d as o3d
import shutil, math

def display_3D(pcd, inliers):
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 1, 0])  # 绿色

    # 非地面点（红色）
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])  # 红色

    # 显示结果
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def set_planes(src_path, target_path,calib_path):
    calib = Calibration(calib_path)
    points = np.fromfile(src_path, dtype=np.float32).reshape(-1, 4)
    points_velo = points[:, :3]
    points_rect = calib.project_velo_to_rect(points_velo)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_rect)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.4,  # 内点距离阈值（单位：米）
        ransac_n=3,  # 每次采样3个点
        num_iterations=10000  # 迭代次数
    )
    a, b, c, d = plane_model
    if b > 0:
        plane_model =  -plane_model
        a, b, c, d = plane_model
    #显示地面分隔
    #display_3D(pcd, inliers)

    with open(target_path, 'w') as f:
        f.write('# Matrix\n')
        f.write('WIDTH 4\n')
        f.write('HEIGHT 1\n')
        f.write(f"{a:.6e} {b:.6e} {c:.6e} {d:.6e}\n")


def chang_data(A_path, B_path):
    # 1. 复制文件 A_path 到 B_path
    shutil.copy2(A_path, B_path)

    # 2. 读取 B_path 文件并按行修改
    with open(B_path, 'r') as file:
        lines = file.readlines()

    # 3. 修改每一行
    modified_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # 如果不是空行
            # 分割行数据
            parts = stripped_line.split()
            if len(parts) >= 4:  # 确保至少有4个数据
                # 计算倒数第二和倒数第三的和
                a11 = float(parts[12]) * -1
                a12 = float(parts[8])* 0.5 - float(parts[13]) + 0.25
                a13 = float(parts[11])-0.2
                a14 = -(float(parts[14]) + np.pi / 2)
                beta = np.arctan2(a13,a11)
                alpha = a14 + beta -np.sign(beta) * np.pi / 2
                new_parts = ['Car','0.00','0', str(alpha)] +parts[4:-7] + [str(float(parts[8])+0.25),str(float(parts[9])+0.4),str(float(parts[10])+0.5),str(a11), str(a12), str(a13), str(a14)]
                modified_line = ' '.join(new_parts) + '\n'
                modified_lines.append(modified_line)

    # 4. 将修改后的内容写回 B_path
    with open(B_path, 'w') as file:
        file.writelines(modified_lines)

def get_train_val_txt_kitti(src_path):
    """
    数据格式:KITTI
    # For KITTI Dataset
    └── KITTI_DATASET_ROOT
        ├── training    <-- 7481 train data
        |   ├── image_2 <-- for visualization
        |   ├── calib
        |   ├── label_2
        |   └── velodyne
        └── testing     <-- 7580 test data
            ├── image_2 <-- for visualization
            ├── calib
            └── velodyne

    src_path: KITTI_DATASET_ROOT kitti文件夹

    """
    # 1.自动生成数据集划分文件夹ImageSets
    set_path = "%s/KITTI/ImageSets/"%src_path
    object_path = "%s/KITTI/object/"%src_path
    if os.path.exists(set_path):  # 如果文件存在
        shutil.rmtree(set_path)  # 清空原始数据
    if os.path.exists(object_path):  # 如果文件存在
        shutil.rmtree(object_path)  # 清空原始数据

    os.makedirs(set_path )
    os.makedirs(object_path + '/testing/calib')
    os.makedirs(object_path + '/testing/velodyne')
    os.makedirs(object_path + '/testing/image_2')
    os.makedirs(object_path + '/training/calib')
    os.makedirs(object_path + '/training/velodyne')
    os.makedirs(object_path + '/training/planes')
    os.makedirs(object_path + '/training/image_2')
    os.makedirs(object_path + '/training/label_2')

    # 2.训练样本分割  生成train.txt val.txt trainval.txt
    train_list = os.listdir(os.path.join(src_path,'labels'))
    #random.shuffle(train_list)  # 打乱顺序，随机采样
    # 设置训练和验证的比例
    train_p = 0.9

    # 开始写入分割文件
    f_train = open(os.path.join(set_path, "train.txt"), 'w')
    f_val = open(os.path.join(set_path, "val.txt"), 'w')
    f_trainval = open(os.path.join(set_path, "trainval.txt"), 'w')
    from tqdm import tqdm
    for i,src in enumerate(tqdm(train_list, desc="Processing planes")):
        shutil.copy2(src_path + '/pointclouds/'+src[:-4]+'.bin', object_path + '/training/velodyne/'+src[:-4]+'.bin')
        shutil.copy2(src_path + '/calib.txt',object_path + '/training/calib/' + src[:-4] + '.txt')
        shutil.copy2(src_path + '/image.png',object_path + '/training/image_2/' + src[:-4] + '.png')
        set_planes(src_path + '/pointclouds/'+src[:-4]+'.bin', object_path + '/training/planes/'+src[:-4]+'.txt',object_path + '/training/calib/' + src[:-4] + '.txt' )
        chang_data(src_path+'/labels/'+src, object_path + '/training/label_2/'+src)
        if i < int(len(train_list) * train_p):
            f_train.write(src[:-4] + '\n')
            f_trainval.write(src[:-4] + '\n')
        else :
            f_val.write(src[:-4] + '\n')
            f_trainval.write(src[:-4] + '\n')
            shutil.copy2(src_path + '/pointclouds/'+src[:-4]+'.bin', object_path + '/testing/velodyne/'+src[:-4]+'.bin')
            shutil.copy2(src_path + '/calib.txt',object_path + '/testing/calib/' + src[:-4] + '.txt')
            shutil.copy2(src_path + '/image.png',object_path + '/testing/image_2/' + src[:-4] + '.png')

    # 3.测试样本分割  生成test.txt
    test_list = os.listdir(os.path.join(object_path,'testing','velodyne'))
    f_test = open(os.path.join(set_path, "test.txt"), 'w')
    for i,src in enumerate(test_list):
        f_test.write(src[:-4] + '\n')

#
# def create_default_calib_file(calib_file_path):
#     """
#     创建默认的KITTI标定文件
#     如果你的数据是纯点云数据，可以使用这些默认值
#     """
#     # 默认的相机内参矩阵 (如果没有相机数据，使用虚拟值)
#     P0 = "P0: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 0.000000000000e+00 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
#     P1 = "P1: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.875744000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
#     P2 = "P2: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 4.485728000000e+01 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.163791000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.745884000000e-03"
#     P3 = "P3: 7.215377000000e+02 0.000000000000e+00 6.095593000000e+02 -3.395242000000e+02 0.000000000000e+00 7.215377000000e+02 1.728540000000e+02 2.199936000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 2.729905000000e-03"
#
#     # 矫正矩阵
#     R0_rect = "R0_rect: 9.999239000000e-01 9.837760000000e-03 -7.445048000000e-03 -9.869795000000e-03 9.999421000000e-01 -4.278459000000e-03 7.402527000000e-03 4.351614000000e-03 9.999631000000e-01"
#
#     # Velodyne到相机的变换矩阵 (如果是纯点云，可以使用单位矩阵)
#     Tr_velo_to_cam = "Tr_velo_to_cam: 7.533745000000e-03 -9.999714000000e-01 -6.166020000000e-04 -4.069766000000e-03 1.480249000000e-02 7.280733000000e-04 -9.998902000000e-01 -7.631618000000e-02 9.998621000000e-01 7.523790000000e-03 1.480755000000e-02 -2.717806000000e-01"
#
#     # IMU到Velodyne的变换矩阵
#     Tr_imu_to_velo = "Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01"
#
#     # 写入标定文件
#     with open(calib_file_path, 'w') as f:
#         f.write(P0 + '\n')
#         f.write(P1 + '\n')
#         f.write(P2 + '\n')
#         f.write(P3 + '\n')
#         f.write(R0_rect + '\n')
#         f.write(Tr_velo_to_cam + '\n')
#         f.write(Tr_imu_to_velo + '\n')
#

def generate_calib_files_for_dataset(data_root, split='train'):
    """
    为整个数据集生成标定文件
    """
    # 读取样本ID列表
    split_file = os.path.join(data_root, 'KITTI', 'ImageSets', f'{split}.txt')
    if not os.path.exists(split_file):
        print(f"Split file not found: {split_file}")
        return

    with open(split_file, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines()]

    # 创建calib文件夹
    calib_dir = os.path.join(data_root, 'KITTI', 'object', split+'ing', 'calib')
    os.makedirs(calib_dir, exist_ok=True)

    # 为每个样本生成标定文件
    for sample_id in sample_ids:
        calib_txt = data_root + '/calib.txt'
        dst_path = os.path.join(calib_dir, f'{int(sample_id):06d}.txt')
        shutil.copy2(calib_txt, dst_path)

        #lib_file = os.path.join(calib_dir, f'{int(sample_id):06d}.txt')
        #create_default_calib_file(calib_file)
        #print(f"Generated: {calib_file}")

    print(f"Generated {len(sample_ids)} calibration files")


if __name__=='__main__':
    """
    	src_path: 数据目录
    """
    src_path = 'G:/3D/labelCloud'
    get_train_val_txt_kitti(src_path)

    generate_calib_files_for_dataset(src_path, split='train')


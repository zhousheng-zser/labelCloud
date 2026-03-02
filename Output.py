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

    a = 0.004234
    b = 0.009570
    c = 0.999945
    d = 3.852301
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
    with open(B_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 3. 修改每一行
    modified_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:  # 如果不是空行
            # 分割行数据 (labelCloud 的格式通常为: class truc occ alpha bbox[4] dim[3] loc[3] ry)
            parts = stripped_line.split()
            if len(parts) >= 15:  # 确保包含足够的 KITTI 字段
                obj_type = parts[0]
                
                # labelCloud 的 3D 尺寸 (通常按照 length, width, height 的顺序保存，这里解析为原版的顺序)
                # 修改前代码将 parts[8,9,10] 错开，标准 labelCloud 导出: parts[8]=height, parts[9]=width, parts[10]=length
                # 我们假设原始值为 h_lc, w_lc, l_lc
                h_lc = float(parts[8])
                w_lc = float(parts[9])
                l_lc = float(parts[10])
                
                # labelCloud 的 3D 中心点 (雷达坐标系 x, y, z)
                x_lc = float(parts[11])
                y_lc = float(parts[12])
                z_lc = float(parts[13])
                
                # labelCloud 的旋转角 (yaw)
                ry_lc = float(parts[14])
                
                # ------------------- 核心转换 -------------------
                
                # 坐标系转换：从 雷达系 (x前, y左, z上) -> 相机系 (x右, y下, z前)
                # 并且中心点从 几何中心 -> 底面中心
                x_cam = -y_lc
                y_cam = -z_lc + (h_lc / 2.0)  # 将中心点移到底面
                z_cam = x_lc
                
                # 旋转角转换 (通常是相反的方向并且加上 pi/2)
                ry_cam = -(ry_lc + np.pi / 2.0)
                
                # 维度赋值 (KITTI 要求输出顺序: height, width, length)
                # 观察到原代码可能将维度混淆，我们强制统一格式:
                h_kitti = h_lc
                w_kitti = w_lc
                l_kitti = l_lc
                
                # 计算 alpha (观察角)
                alpha = ry_cam - np.arctan2(-x_cam, z_cam)
                
                # -----------------------------------------------
                
                # 构造符合 KITTI label_2 格式的 15 个字段
                # 0: type
                # 1: truncated (default: 0.00)
                # 2: occluded (default: 0)
                # 3: alpha
                # 4-7: 2D bbox (left, top, right, bottom) 保留原有数据
                # 8-10: 3D dims (height, width, length)
                # 11-13: 3D loc (x, y, z) in cam coord
                # 14: rotation_y in cam coord
                new_parts = [
                    obj_type,
                    '0.00',
                    '0',
                    f"{alpha:.4f}",
                ] + parts[4:8] + [
                    f"{h_kitti:.4f}",
                    f"{w_kitti:.4f}",
                    f"{l_kitti:.4f}",
                    f"{x_cam:.4f}",
                    f"{y_cam:.4f}",
                    f"{z_cam:.4f}",
                    f"{ry_cam:.4f}"
                ]
                
                modified_line = ' '.join(new_parts) + '\n'
                modified_lines.append(modified_line)

    # 4. 将修改后的内容写回 B_path
    with open(B_path, 'w', encoding='utf-8') as file:
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
    train_p = 1
    #train_p = 0.9

    # 开始写入分割文件
    f_train = open(os.path.join(set_path, "train.txt"), 'w')
    f_val = open(os.path.join(set_path, "val.txt"), 'w')
    f_trainval = open(os.path.join(set_path, "trainval.txt"), 'w')
    from tqdm import tqdm
    for i,src in enumerate(tqdm(train_list, desc="Processing planes")):
        shutil.copy2(src_path + '/pointclouds/'+src[:-4]+'.bin', object_path + '/training/velodyne/'+src[:-4]+'.bin')
        shutil.copy2(src_path + '/calib.txt',object_path + '/training/calib/' + src[:-4] + '.txt')
        shutil.copy2(src_path + '/image.png',object_path + '/training/image_2/' + src[:-4] + '.png')
        shutil.copy2(src_path + '/planes/'+src[:-4]+'.txt',object_path + '/training/planes/' + src[:-4] + '.txt')
        #set_planes(src_path + '/pointclouds/'+src[:-4]+'.bin', object_path + '/training/planes/'+src[:-4]+'.txt',object_path + '/training/calib/' + src[:-4] + '.txt' )
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


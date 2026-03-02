#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI 点云 + 标签 + 标定 -> 3D 框 PCD 转换工具 (Windows 兼容 + 独立运行版)

示例 (Windows PowerShell):
    python bin_label_to_pcd.py ^
        --lidar G:/3D/labelCloud/KITTI/object/training/velodyne/022701.bin ^
        --label G:/3D/labelCloud/KITTI/object/training/label_2/022701.txt ^
        --calib G:/3D/labelCloud/KITTI/object/training/calib/022701.txt ^
        --output ./022701.pcd
"""

import argparse
from pathlib import Path
import numpy as np

# 定义框的边连接关系 (8个角点连接成12条边)
EDGE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
]

# --- 独立实现的工具类与函数，解决 Windows 上难以安装 OpenPCDet 的问题 ---

class Calibration:
    """
    负责读取 KITTI 标定文件并进行坐标转换
    """
    def __init__(self, calib_file):
        if not Path(calib_file).exists():
            raise FileNotFoundError(f"标定文件未找到: {calib_file}")
            
        lines = open(calib_file, 'r', encoding='utf-8').readlines()
        data = {}
        for line in lines:
            if not line.strip() or ':' not in line:
                continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
        
        # P2: 矩形相机2的投影矩阵 (3, 4)
        self.P2 = data['P2'].reshape(3, 4)
        # R0: 矫正旋转矩阵 (3, 3)
        self.R0 = data['R0_rect'].reshape(3, 3)
        # V2C: Velodyne 到相机的平移旋转矩阵 (3, 4)
        self.V2C = data['Tr_velo_to_cam'].reshape(3, 4)

    def rect_to_velo(self, pts_rect):
        """ 相机矩形坐标系 -> 雷达坐标系 """
        # pts_ref = pts_rect * inv(R0)
        pts_ref = np.dot(pts_rect, np.linalg.inv(self.R0).T)
        
        # 扩展 V2C 为 4x4 矩阵
        V2C_4x4 = np.eye(4)
        V2C_4x4[:3, :] = self.V2C
        C2V = np.linalg.inv(V2C_4x4)
        
        pts_ref_hom = np.hstack((pts_ref, np.ones((pts_ref.shape[0], 1))))
        pts_velo = np.dot(pts_ref_hom, C2V.T)
        return pts_velo[:, :3]

def boxes3d_camera_to_lidar(boxes3d_camera, calib):
    """
    将相机坐标系下的框转换到雷达坐标系
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, ry]
    Returns:
        boxes3d_lidar: (N, 7) [x, y, z, l, w, h, ry]
    """
    xyz_camera = boxes3d_camera[:, 0:3]
    xyz_lidar = calib.rect_to_velo(xyz_camera)
    
    l, h, w, r = boxes3d_camera[:, 3], boxes3d_camera[:, 4], boxes3d_camera[:, 5], boxes3d_camera[:, 6]
    
    # 角度转换
    r_lidar = -r - np.pi / 2
    r_lidar = (r_lidar + np.pi) % (2 * np.pi) - np.pi
    
    # 结果格式为 [x, y, z, l, w, h, r]
    return np.concatenate([xyz_lidar, l[:, None], w[:, None], h[:, None], r_lidar[:, None]], axis=1)

def boxes_to_corners_3d(boxes3d):
    """
    雷达坐标系下的框转为8个角点坐标
    Args:
        boxes3d: (N, 7) [x, y, z, l, w, h, ry]
    """
    # 基础模板：中心在(0,0,0) 的单位立方体
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2

    corners3d = np.tile(template[np.newaxis, :, :], (boxes3d.shape[0], 1, 1))
    # 缩放
    corners3d[:, :, 0] *= boxes3d[:, 3, np.newaxis] # l
    corners3d[:, :, 1] *= boxes3d[:, 4, np.newaxis] # w
    corners3d[:, :, 2] *= boxes3d[:, 5, np.newaxis] # h

    # 围绕 Z 轴旋转
    cosa = np.cos(boxes3d[:, 6])
    sina = np.sin(boxes3d[:, 6])
    for i in range(len(boxes3d)):
        rot_mat = np.array([
            [cosa[i], -sina[i], 0],
            [sina[i], cosa[i], 0],
            [0, 0, 1]
        ])
        corners3d[i] = np.dot(corners3d[i], rot_mat.T)

    # 坐标平移
    # KITTI 的 (x,y,z) 是底面中心，所以我们的 Z 轴投影需要向上平移 h/2
    corners3d[:, :, 0] += boxes3d[:, 0, np.newaxis]
    corners3d[:, :, 1] += boxes3d[:, 1, np.newaxis]
    corners3d[:, :, 2] += (boxes3d[:, 2, np.newaxis] + boxes3d[:, 5, np.newaxis] / 2)
    
    return corners3d

# --- 业务逻辑 ---

def parse_args():
    parser = argparse.ArgumentParser(description="KITTI 点云加框导出 PCD")
    parser.add_argument("--lidar", required=True, help="velodyne/xxxxxx.bin 路径")
    parser.add_argument("--label", required=True, help="label_2/xxxxxx.txt 路径")
    parser.add_argument("--calib", required=True, help="calib/xxxxxx.txt 路径")
    parser.add_argument("--output", default=None, help="输出 PCD 文件路径（默认同文件名）")
    parser.add_argument("--edge-samples", type=int, default=20, help="每条边采样点数")
    parser.add_argument("--ascii", action="store_true", help="使用 ASCII 方式保存（默认二进制）")
    return parser.parse_args()

def load_lidar(bin_path: str) -> np.ndarray:
    """ 读取雷达点云二进制文件 """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def load_labels(label_path: str) -> np.ndarray:
    """ 读取标签文件 """
    boxes = []
    with open(label_path, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "DontCare":
                continue
            # KITTI 格式: h, w, l, x, y, z, ry (相机坐标系)
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            boxes.append([x, y, z, l, h, w, ry])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 7), dtype=np.float32)

def boxes_to_edge_points(boxes_lidar: np.ndarray, samples_per_edge: int) -> np.ndarray:
    """ 将框的12条边生成采样点，以便在点云中可视化 """
    if boxes_lidar.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    corners = boxes_to_corners_3d(boxes_lidar)
    edge_points = []
    for box_corners in corners:
        for start_idx, end_idx in EDGE_PAIRS:
            start = box_corners[start_idx]
            end = box_corners[end_idx]
            # 在边上均匀采样点
            ts = np.linspace(0.0, 1.0, samples_per_edge, dtype=np.float32)
            edge_points.append(start + np.outer(ts, end - start))
    if edge_points:
        return np.concatenate(edge_points, axis=0)
    return np.zeros((0, 3), dtype=np.float32)

def pack_rgb_from_unit_colors(colors: np.ndarray) -> np.ndarray:
    """ 打包装 RGB 色彩为 PCD 兼容的 float 格式 """
    if colors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    colors_uint = (np.clip(colors, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint32)
    rgb_uint32 = (colors_uint[:, 0] << 16) | (colors_uint[:, 1] << 8) | colors_uint[:, 2]
    return rgb_uint32.view(np.float32)

def write_pcd(points: np.ndarray, colors: np.ndarray, output_path: Path, ascii: bool = False) -> None:
    """ 写入 PCD 文件 """
    if points.shape[0] == 0:
        raise ValueError("No points to write")

    points_f32 = points.astype(np.float32, copy=False)
    colors_f32 = pack_rgb_from_unit_colors(colors)
    point_cloud = np.column_stack((points_f32, colors_f32)).astype(np.float32, copy=False)

    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {point_cloud.shape[0]}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {point_cloud.shape[0]}",
        "DATA ascii" if ascii else "DATA binary",
        ""
    ])

    mode = "w" if ascii else "wb"
    # Windows 下写文本文件建议指定 newline='\n' 防止生成多余的 \r
    with open(output_path, mode, newline='\n' if ascii else None) as f:
        f.write(header if ascii else header.encode("ascii"))
        if ascii:
            np.savetxt(f, point_cloud, fmt="%.8f %.8f %.8f %.8f")
        else:
            f.write(point_cloud.tobytes())

def main():
    args = parse_args()
    lidar_path = Path(args.lidar)
    label_path = Path(args.label)
    calib_path = Path(args.calib)

    for p in [lidar_path, label_path, calib_path]:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    # 加载点云
    points = load_lidar(str(lidar_path))
    # 加载相机坐标系标签
    boxes_cam = load_labels(str(label_path))
    # 加载标定参数
    calib = Calibration(str(calib_path))
    
    box_points = np.zeros((0, 3), dtype=np.float32)
    if boxes_cam.shape[0] > 0:
        # 核心转换：相机坐标 -> 雷达坐标
        boxes_lidar = boxes3d_camera_to_lidar(boxes_cam, calib)
        # 核心转换：框 -> 可视化采样点
        box_points = boxes_to_edge_points(boxes_lidar, args.edge_samples)

    # 合并原始点云和框的采样点
    all_points = points if box_points.shape[0] == 0 else np.vstack((points, box_points))

    # 设置颜色：灰色代表点云，红色代表框
    colors = np.ones_like(all_points) * 0.6
    colors[:points.shape[0]] = [0.7, 0.7, 0.7]
    if box_points.shape[0] > 0:
        colors[points.shape[0]:] = [1.0, 0.0, 0.0]

    # 保存
    output_path = Path(args.output) if args.output else lidar_path.with_suffix(".pcd")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_pcd(all_points, colors, output_path, ascii=args.ascii)
    print(f"成功导出 PCD (包含 3D 框) 至: {output_path}")

if __name__ == "__main__":
    main()

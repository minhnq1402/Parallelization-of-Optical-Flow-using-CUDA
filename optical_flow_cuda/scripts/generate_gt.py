# scripts/generate_gt.py
import numpy as np
from scipy import ndimage
import os

# 1. Tạo thư mục data nếu chưa có
os.makedirs('data', exist_ok=True)

# 2. Khởi tạo 2 khung hình ngẫu nhiên (size nhỏ 100x100 để test nhanh)
width, height = 100, 100
frame1 = np.random.rand(height, width).astype(np.float32)
frame2 = np.random.rand(height, width).astype(np.float32)

# 3. Định nghĩa ma trận Sobel chuẩn xác như trong CUDA
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# 4. Tính toán Ground Truth bằng CPU (Dùng mode='constant' cval=0.0 để xử lý viền giống CUDA)
Ix_gt = ndimage.correlate(frame1, kernel_x, mode='constant', cval=0.0)
Iy_gt = ndimage.correlate(frame1, kernel_y, mode='constant', cval=0.0)
It_gt = frame2 - frame1

# 5. Xuất ra file nhị phân (Binary) để C++ đọc cho lẹ
frame1.tofile('data/frame1.bin')
frame2.tofile('data/frame2.bin')
Ix_gt.tofile('data/Ix_gt.bin')
Iy_gt.tofile('data/Iy_gt.bin')
It_gt.tofile('data/It_gt.bin')

print("[Python] Da tao xong du lieu Ground Truth tai thu muc /data/")
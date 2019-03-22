# upsmapleing sprse point cloud with a RGB image using Markov Random Field

#### 项目概述
+ 在KITTI数据集下使用激光雷达获得的点云与对应的RGB图像对深度进行上采样，得到与RGB相同像素的深度图


#### 依赖

+ OpenCV
+ libpcl
+ Eigen
+ cmake

#### 构建

1. mkdir build
2. cd build
3. cmake -DCMAKE_BUILD_TYPE=Release ..
4. make -j4
5. ./main /home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/velodyne_points/data/0000000100.bin /home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/image_02/data/0000000100.png
# upsampling

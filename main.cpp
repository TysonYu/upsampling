//----- create by tiezheng yu on 2019-2-21-----
//----- useage: ./main /home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/velodyne_points/data/0000000100.bin /home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/image_02/data/0000000100.png
//


#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>
#include <vector>  
#include <string>
#include <math.h>
#include <time.h>

using namespace pcl;
using namespace std;
using namespace cv;
using namespace Eigen;

std::vector<Eigen::Matrix4f> RT;
std::vector<Eigen::Matrix4f> INV;

pcl::PointCloud<pcl::PointXYZI>::Ptr readKittiPclBinData(std::string &in_file)
{
    // load point cloud
    std::fstream input(in_file.c_str(), std::ios::in | std::ios::binary);
    if(!input.good())
    {
        std::cerr << "Could not read file: " << in_file << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);
    pcl::PointCloud<pcl::PointXYZI>::Ptr points (new pcl::PointCloud<pcl::PointXYZI>);
    for (int i=0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        input.read((char *) &point.x, 3*sizeof(float));
        input.read((char *) &point.intensity, sizeof(float));
        points->push_back(point);
    }
    input.close();
    return points;
}

cv::Mat getDepthMap_raw (pcl::PointCloud<pcl::PointXYZ>::Ptr image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera_filtered, cv::Mat I, int max_depth)
{
    Mat M(I.size().height,I.size().width,CV_8UC1);//把点投影到M上
    // namedWindow("Test", CV_WINDOW_AUTOSIZE); 
	//遍历所有像素，初始化像素值
    MatIterator_<uchar>Mbegin,Mend;
	for (Mbegin=M.begin<uchar>(),Mend=M.end<uchar>();Mbegin!=Mend;++Mbegin)
		*Mbegin=255;
	for(int i=0;i<(int)image->points.size();i++)//把深度值投影到图像M上
    {
        if(image->points[i].x>=0  && image->points[i].x<I.size().width && image->points[i].y>=0 && image->points[i].y<I.size().height)
        {
            M.at<uchar>(image->points[i].y,image->points[i].x) = cloud_camera_filtered->points[i].z*255/max_depth;
        }
    }
	// imshow("Test",M);   //窗口中显示图像
    // waitKey(0);
    return M;
}

cv::Mat getDepthMap_kuozhan (pcl::PointCloud<pcl::PointXYZ>::Ptr image, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera_filtered, cv::Mat I, int max_depth)
{
    Mat M(I.size().height,I.size().width,CV_8UC1);//把点投影到M上
    Mat P(I.size().height,I.size().width,CV_8UC1);//扩展投影点
	// namedWindow("Test", CV_WINDOW_AUTOSIZE); 
	//遍历所有像素，初始化像素值
    MatIterator_<uchar>Mbegin,Mend;
	for (Mbegin=M.begin<uchar>(),Mend=M.end<uchar>();Mbegin!=Mend;++Mbegin)
		*Mbegin=255;
	for(int i=0;i<(int)image->points.size();i++)//把深度值投影到图像M上
    {
        if(image->points[i].x>=0  && image->points[i].x<I.size().width && image->points[i].y>=0 && image->points[i].y<I.size().height)
        {
            M.at<uchar>(image->points[i].y,image->points[i].x) = cloud_camera_filtered->points[i].z*255/max_depth;
        }
    }
    for(int count = 0; count < 10; count ++)
    {
        if (count%2 == 0) 
        {
            for (int i=1;i<M.rows-1;i++)
	        {
		        for (int j=1;j<M.cols-1;j++)
		        {
                    if(M.at<uchar>(i,j) == 255)
                    {
                        int temp = 255;
                        int sum = 0;
                        int cnt = 0;
                        for(int n = i-1; n < i+2; n++)
                        {
                            for(int m = j-1; m < j+2; m++)
                            {
                                if(M.at<uchar>(n,m) < temp)
                                {
                                    sum = sum + M.at<uchar>(n,m);
                                    cnt ++;
                                    temp = M.at<uchar>(n,m);
                                }   
                            }
                        }
                        if (cnt > 0) 
                            temp = sum / cnt;
                        P.at<uchar>(i,j) = temp;
                    }
                    else
                        P.at<uchar>(i,j)  = M.at<uchar>(i,j);
		        }
            }
        }
        else
        {
            for (int i=1;i<M.rows-1;i++)
	        {
		        for (int j=1;j<M.cols-1;j++)
		        {
                    if(P.at<uchar>(i,j) == 255)
                    {
                        int sum = 0;
                        int cnt = 0;
                        int temp = 255;
                        for(int n = i-1; n < i+2; n++)
                        {
                            for(int m = j-1; m < j+2; m++)
                            {
                                if(P.at<uchar>(n,m) < temp)
                                {
                                    sum = sum + P.at<uchar>(n,m);
                                    cnt ++;
                                    temp = P.at<uchar>(n,m);
                                }
                            }
                        }
                        if (cnt > 0) 
                            temp = sum / cnt;
                        M.at<uchar>(i,j) = temp;
                    }
                    else
                        M.at<uchar>(i,j)  = P.at<uchar>(i,j);
		        }
            }
        }
    }
	// imshow("Test",M);   //窗口中显示图像
    // waitKey(0);
    return M;
}

int main(int argc, char **argv)
{
    //---------读取并转化点云为pcl(存在cloud)--------------------------------------------------------------

    clock_t start = clock();
    // string in_file = "/home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/velodyne_points/data/0000000100.bin";
    // string img = "/home/icey/Desktop/project/KITTI/2011_09_26_drive_0005_sync/image_02/data/0000000100.png";
    string in_file = argv[1];
    string img = argv[2];
    pcl::PointCloud<pcl::PointXYZI>::Ptr original_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr image (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud_world (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr image_result (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result_xyz_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_rgb_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    original_cloud = readKittiPclBinData(in_file);//把bin转化为pcd
    pcl::copyPointCloud(*original_cloud, *cloud);//把PointXYZI转化为PointXYZ
    // pcl::visualization::PCLVisualizer viewer("original Cloud viewer");
    // viewer.addPointCloud(cloud, "sample cloud");
    // viewer.setBackgroundColor(0,0,0);
    // viewer.addCoordinateSystem();
    // while(!viewer.wasStopped())
    //     //while (!viewer->wasStopped ())
    //     viewer.spinOnce();
    
    //---------转化到相机坐标系---------------------------------------------------------------------------
    Eigen::Matrix4f rt1;//激光雷达到相机cam0的RT矩阵
    Eigen::Matrix4f rt2;//cam0-to-cam2;
    Eigen::Matrix4f rt;
    Eigen::Matrix4f inv;
    rt1 << 7.533745e-03,-9.999714e-01,-6.166020e-04,-4.069766e-03,   1.480249e-02,7.280733e-04,-9.998902e-01,-7.631618e-02,  9.998621e-01,7.523790e-03,1.480755e-02,-2.717806e-01, 0,0,0,1;  
    rt2 << 9.999758e-01, -5.267463e-03,-4.552439e-03,5.956621e-02,    5.251945e-03,9.999804e-01,-3.413835e-03,2.900141e-04, 4.570332e-03,3.389843e-03,9.999838e-01,2.577209e-03, 0,0,0,1;
    rt = rt2*rt1;
    // rt << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03, -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02, 9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01, 0,0,0,1;
    inv=rt.inverse();
    RT.push_back(rt);
    INV.push_back(inv);
    pcl::transformPointCloud (*cloud, *cloud_camera, RT[0]);//cloud_camera相机坐标系
    //---------过滤掉相机坐标系下z小于0的点云--------------------------------------------------------------
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_camera);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 500);//delete all the point that z<0 && z>40
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_camera_filtered);
    //---------转化到图像平面上--------------------------------------------------------------------------
    cv::Mat I = imread(img, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(I, I, CV_RGBA2RGB);
    Mat_<Vec3b> _I = I;
    Eigen::Matrix4f intrisic;//相机内参
    intrisic << 7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03,    0,0,0,1;
    // intrisic << 7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 4.688783000000e+01, 0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03, 0,0,0,1;
    pcl::transformPointCloud (*cloud_camera_filtered, *image, intrisic);//image，Z上未归一化的像素坐标系
    for(int i = 0; i < (int)image->points.size(); i++)
    {
        image->points[i].x = image->points[i].x / image->points[i].z;
        image->points[i].y = image->points[i].y / image->points[i].z;
    }
    int max_depth = 0;
    for(int i=0;i<(int)image->points.size();i++)
    {
        if(image->points[i].x>=0  && image->points[i].x<1242 && image->points[i].y>=0 && image->points[i].y<375)
        {
            pcl::PointXYZRGB point;
            point.x = cloud_camera_filtered->points[i].x;
            point.y = cloud_camera_filtered->points[i].y;
            point.z = cloud_camera_filtered->points[i].z;
            point.r = _I(round(image->points[i].y),round(image->points[i].x))[2];//round() most closed number
            point.g = _I(round(image->points[i].y),round(image->points[i].x))[1];
            point.b = _I(round(image->points[i].y),round(image->points[i].x))[0];
            //---------实现rgb颜色渐变----------------------------------------------------------
            // cv::Point add_point;//特征点，用以画在图像中 
            // int r=0,g=0,b=0; 
	        // add_point.x = round(image->points[i].x);//特征点在图像中横坐标  
	        // add_point.y = round(image->points[i].y);//特征点在图像中纵坐标
            // int val = cloud_camera_filtered->points[i].z;
            // if (val < 22)//第一个三等分
            // {
            //     r = (int)(one * val);
            //     g = 255;
            // }
            // else if (val >= 22 && val < 44)//第二个三等分
            // {
            //     r = 255;
            //     g = 255 - (int)((val - 22) * one);//val减最大取值的三分之一
            // }
            // else { r = 255; }//最后一个三等分
	        // cv::circle(I, add_point, 1, cv::Scalar(r, g, b));//在图像中画出特征点，2是圆的半径
            //---------找到图像中的深度最大值-----------------------------------------------------
            if (cloud_camera_filtered->points[i].z > max_depth)
                max_depth = cloud_camera_filtered->points[i].z;
            colored_cloud->points.push_back(point);
        }
    }
    //---------创建初始深度图像------------------------------------------------------------------------------
    Mat inil_depth_map = getDepthMap_kuozhan (image, cloud_camera_filtered, I, max_depth);//使用多次迭代获取每一个点的初始深度值
    //---------在图像中取一小块做实验--------------------------------------------------------------------- 
    Mat raw_depth_map = getDepthMap_raw (image, cloud_camera_filtered, I, max_depth);
    Rect rect(0, 150, 1000, 200);
    Mat small_RGB_image = I(rect);//RGB图像中的一小块
    Mat_<Vec3b> _small_RGB_image = small_RGB_image;
    Mat small_depth_image = inil_depth_map(rect);//用临近深度填充的深度图中的一小块
    Mat small_gray_image = raw_depth_map(rect);//初始稀疏深度图中的一小块
    Mat result(small_gray_image.size().height,small_gray_image.size().width,CV_8UC1);//result 用于储存结果
    #define TOTAL 1000*200
    #define TOTAL_16 1000*200*16
    Eigen::SparseMatrix < double > A_1 (TOTAL , TOTAL) ;//创建一个稀疏矩阵A_1
    Eigen::SparseMatrix < double > A_2 (TOTAL , TOTAL) ;//创建一个稀疏矩阵A_2
    Eigen::SparseMatrix < double > A (TOTAL , TOTAL) ;//创建一个稀疏矩阵A
    // Eigen::SparseMatrix < double > Big (33000 , 33000) ;//创建一个稀疏矩阵A
    Eigen::SparseMatrix < double > b (TOTAL , 1) ;//创建一个稀疏矩阵b
    Eigen::VectorXd x (TOTAL) ;//创建向量储存结果。
    // int k = 1, c = 5;
    std::vector < Eigen::Triplet < double > > triplets_A_1 ;//创建一个用于初始化稀疏矩阵的向量A_1
    std::vector < Eigen::Triplet < double > > triplets_b ;//创建一个用于初始化稀疏矩阵的向量b
    std::vector < Eigen::Triplet < double > > triplets_A_2 ;//创建一个用于初始化稀疏矩阵的向量A_2
    std::vector < Eigen::Triplet < double > > triplets_A ;//创建一个用于初始化稀疏矩阵的向量A_2
    int total_pix = small_gray_image.size().height * small_gray_image.size().width;
    cout << "total_pix = " << total_pix << endl;
    // triplets.reserve(estimation_of_entries);
    // int r[TOTAL_16];// 非零元素的行号
    // int c[TOTAL_16];// 非零元素的列号
    // double val[TOTAL_16];// 非零元素的值
    int *r = new int[TOTAL_16];
    int *c = new int[TOTAL_16];
    double *val = new double[TOTAL_16];
    for(int i = 0; i < total_pix; i++)
    {
        r[i] = i;
        c[i] = i;
        val[i] = 1;
    }
    for ( int i = 0 ; i < total_pix ; i++ )
         triplets_A_1.push_back ( Eigen::Triplet < double >(r[i] , c[i] , val[i]) );
    A_1.setFromTriplets ( triplets_A_1.begin ( ) , triplets_A_1.end ( ) );// 初始化A_1
    for(int  i = 0; i < total_pix; i++)
    {
        val[i] = 0;
        c[i] = 0;
        // if(small_depth_image.at<uchar>(i/small_gray_image.size().width,i%small_gray_image.size().width) != 255)
            val[i] =  small_depth_image.at<uchar>(i/small_gray_image.size().width,i%small_gray_image.size().width);
        // cout << "r = " << r[i] << " c = " << c[i] << " val = " << val[i] << endl;
    }
    for ( int i = 0 ; i < total_pix ; i++ )
         triplets_b.push_back ( Eigen::Triplet < double >(r[i] , c[i] , val[i]) );
    b.setFromTriplets ( triplets_b.begin ( ) , triplets_b.end ( ) );// 初始化b
    long flag = 0;
    double C = 1;
    for(int i = 0; i < total_pix; i++ )
    {
        if (i == 0)//左上角
        {
            int j_1 = i + 1;
            int j_2 = i + small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v + 1;//右边邻居
            int u_2 = u + 1;//下边邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j_1;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = j_1;
            val[flag] = w_1;
            flag ++;
            double w_2 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_2,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_2,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_2,v)[0],2)));
            r[flag] = i;
            c[flag] = j_2;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = i;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = j_2;
            val[flag] = w_2;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1 + w_2;
            flag ++;
        }
        else if (i == small_gray_image.size().width)//右上角
        {
            int j_1 = i - 1;
            int j_2 = i + small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v - 1;//左邻居
            int u_2 = u + 1;//下邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j_1;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = j_1;
            val[flag] = w_1;
            flag ++;
            double w_2 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_2,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_2,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_2,v)[0],2)));
            r[flag] = i;
            c[flag] = j_2;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = i;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = j_2;
            val[flag] = w_2;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1 + w_2;
            flag ++;
        }
        else if(i == total_pix - small_gray_image.size().width)//左下角
        {
            int j_1 = i - small_gray_image.size().width;//
            int j_2 = i + 1;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v + 1;//右邻居
            int u_2 = u - 1;//上邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j_1;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = j_1;
            val[flag] = w_1;
            flag ++;
            double w_2 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_2,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_2,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_2,v)[0],2)));
            r[flag] = i;
            c[flag] = j_2;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = i;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = j_2;
            val[flag] = w_2;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1 + w_2;
            flag ++;
        }
        else if (i == -1)//右下角
        {
            int j_1 = i - 1;
            int j_2 = i - small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v - 1;//左邻居
            int u_2 = u - 1;//上邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j_1;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = j_1;
            val[flag] = w_1;
            flag ++;
            double w_2 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_2,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_2,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_2,v)[0],2)));
            r[flag] = i;
            c[flag] = j_2;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = i;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = j_2;
            val[flag] = w_2;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1 + w_2;
            flag ++;
        }
        else if (i >= 1 && i <= small_gray_image.size().width-2)//上排
        {
            int j = i + small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int u_1 = u + 1;//下邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_1,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_1,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_1,v)[0],2)));
            r[flag] = i;
            c[flag] = j;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = j;
            val[flag] = w_1;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1;
            flag ++;
        }
        else if (i>0 && i%small_gray_image.size().width == 0)//左排
        {
            int j = i + 1;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v + 1;//右邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = j;
            val[flag] = w_1;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1;
            flag ++;
        }
        else if (i>0 && i%small_gray_image.size().width == small_gray_image.size().width-1)//右排
        {
            int j = i - 1;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v - 1;//左邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = j;
            val[flag] = w_1;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1;
            flag ++;
        }
        else if (i > total_pix - small_gray_image.size().width && i < total_pix-1)//下排
        {
            int j = i - small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int u_1 = u - 1;//上邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_1,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_1,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_1,v)[0],2)));
            r[flag] = i;
            c[flag] = j;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j;
            c[flag] = j;
            val[flag] = w_1;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1;
            flag ++;
        }
        else//中间
        {
            int j_1 = i - 1;
            int j_2 = i + 1;
            int j_3 = i - small_gray_image.size().width;
            int j_4 = i + small_gray_image.size().width;
            int u = i / small_gray_image.size().width;//y坐标
            int v = i % small_gray_image.size().width;//x坐标
            int v_1 = v - 1;//左邻居
            int v_2 = v + 1;//右邻居
            int u_3 = u - 1;//上邻居
            int u_4 = u + 1;//下邻居
            double w_1 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_1)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_1)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_1)[0],2)));
            r[flag] = i;
            c[flag] = j_1;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = i;
            val[flag] = -w_1;
            flag ++;
            r[flag] = j_1;
            c[flag] = j_1;
            val[flag] = w_1;
            flag ++;
            double w_2 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u,v_2)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u,v_2)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u,v_2)[0],2)));
            r[flag] = i;
            c[flag] = j_2;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = i;
            val[flag] = -w_2;
            flag ++;
            r[flag] = j_2;
            c[flag] = j_2;
            val[flag] = w_2;
            flag ++;
            double w_3 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_3,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_3,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_3,v)[0],2)));
            r[flag] = i;
            c[flag] = j_3;
            val[flag] = -w_3;
            flag ++;
            r[flag] = j_3;
            c[flag] = i;
            val[flag] = -w_3;
            flag ++;
            r[flag] = j_3;
            c[flag] = j_3;
            val[flag] = w_3;
            flag ++;
            double w_4 = exp(-C*sqrt(pow(_small_RGB_image(u,v)[2] - _small_RGB_image(u_4,v)[2],2) + pow(_small_RGB_image(u,v)[1] - _small_RGB_image(u_4,v)[1],2) + pow(_small_RGB_image(u,v)[0] - _small_RGB_image(u_4,v)[0],2)));
            r[flag] = i;
            c[flag] = j_4;
            val[flag] = -w_4;
            flag ++;
            r[flag] = j_4;
            c[flag] = i;
            val[flag] = -w_4;
            flag ++;
            r[flag] = j_4;
            c[flag] = j_4;
            val[flag] = w_4;
            flag ++;
            r[flag] = i;
            c[flag] = i;
            val[flag] = w_1 + w_2 + w_3 + w_4;
            flag ++;
        }
    }
    for ( int i = 0 ; i < flag ; i++ )
         triplets_A_2.push_back ( Eigen::Triplet < double >(r[i] , c[i] , val[i]) );
    A_2.setFromTriplets ( triplets_A_2.begin ( ) , triplets_A_2.end ( ) );// 初始化A_1
    A = A_1 + A_2;
    // cout << A << endl;
    //----------共轭梯度求解矩阵----------------------------------------------------------------------
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
    cg.compute(A);
    x = cg.solve(b);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;
    // cout << x << endl;
    //---------用解得的y对深度进行赋值------------------------------------------------------------------
    for(int i = 0; i < TOTAL; i++)
    {
        if (x(i) >= 255)
            result.at<uchar>(i/result.size().width,i%result.size().width) = 255;
        else
            result.at<uchar>(i/result.size().width,i%result.size().width) = x(i);
    }
    //---------把深度图返回到点云-----------------------------------------------------------------------
    for(int i = 0; i < TOTAL; i++)
    {
        pcl::PointXYZ point;
        point.z = x(i)/255*max_depth;
        point.x = i%result.size().width * point.z;
        point.y = i/result.size().width * point.z;
        image_result->points.push_back(point);
    }
    Eigen::Matrix4f intrisic_inv = intrisic.inverse();
    pcl::transformPointCloud (*image_result, *result_xyz_cloud, intrisic_inv);//image，Z上未归一化的像素坐标系
    for(int i = 0; i < TOTAL; i++)
    {
        pcl::PointXYZRGB point;
        point.z = result_xyz_cloud->points[i].z;
        point.x = result_xyz_cloud->points[i].x;
        point.y = result_xyz_cloud->points[i].y;
        point.r = _small_RGB_image(round(i/result.size().width),round(i%result.size().width))[2];//round() most closed number
        point.g = _small_RGB_image(round(i/result.size().width),round(i%result.size().width))[1];
        point.b = _small_RGB_image(round(i/result.size().width),round(i%result.size().width))[0];
        result_rgb_cloud->points.push_back(point);
    }
    result_rgb_cloud->width = 1;
    result_rgb_cloud->height = result_rgb_cloud->points.size();
    pcl::io::savePCDFileASCII ("../result/result.pcd", *result_rgb_cloud);
    // cv::imwrite("rgb_image.jpg",small_RGB_image);
    // colored_cloud->width = 1;
    // colored_cloud->height = colored_cloud->points.size();
    // pcl::io::savePCDFileASCII ("../result/colored_cloud.pcd", *colored_cloud);
    
    // cout << b << endl;
    // cv::namedWindow("raw depth map", CV_WINDOW_AUTOSIZE);
    // cv::imshow("raw depth map",raw_depth_map);
    // waitKey(0);
    // cv::namedWindow("depth map", CV_WINDOW_AUTOSIZE);
    // cv::imshow("depth map",inil_depth_map);
    // waitKey(0);
    // cv::namedWindow("small rgb map", CV_WINDOW_AUTOSIZE);
    // cv::imshow("small rgb map",small_RGB_image);
    // waitKey(0);
    // cv::namedWindow("small depth map", CV_WINDOW_AUTOSIZE);
    // cv::imshow("small depth map",small_depth_image);
    // waitKey(0);
    // cv::namedWindow("small gray map", CV_WINDOW_AUTOSIZE);
    // cv::imshow("small gray map",small_gray_image);
    // waitKey(0);
    // cv::namedWindow("result", CV_WINDOW_AUTOSIZE);
    // cv::imshow("result",result);
    // waitKey(0);
    //---------把上色的点云重新转化到世界坐标系-------------------------------------------------------------
    // pcl::transformPointCloud (*colored_cloud, *colored_cloud_world, INV[0]);
    clock_t end = clock();
    cout << "cost time = " << (double)(end-start)/CLOCKS_PER_SEC <<  " s"<< endl;
    // pcl::visualization::PCLVisualizer camera_color_viewer("Camera Cloud viewer");
    // camera_color_viewer.addPointCloud(colored_cloud, "sample cloud");
    // camera_color_viewer.setBackgroundColor(0,0,0);
    // camera_color_viewer.addCoordinateSystem();
    // while(!camera_color_viewer.wasStopped())
    //     camera_color_viewer.spinOnce();
    // pcl::visualization::PCLVisualizer result_viewer("result");
    // result_viewer.addPointCloud(result_rgb_cloud, "sample cloud");
    // result_viewer.setBackgroundColor(0,0,0);
    // result_viewer.addCoordinateSystem();
    // while(!result_viewer.wasStopped())
    //     result_viewer.spinOnce();
    delete [] r;
    delete [] c;
    delete [] val;
    return 0;
}

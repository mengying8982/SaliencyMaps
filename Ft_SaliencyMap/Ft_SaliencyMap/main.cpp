//
//  main.cpp
//  Ft_SaliencyMap
//
//  Created by 孟莹 on 16/4/22.
//  Copyright © 2016年 孟莹. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv/highgui.h>
#include <opencv/cv.h>

using namespace std;
//-------------------------------------------
//imshow函数，用来显示名为TEST的函数，按任意键继续
//-------------------------------------------

void imshow(const cv::Mat &image)
{
    cv::imshow("TEST", image);
    cv::waitKey(0);
}

//将图片分离出rgb三通道，每一通道求平均值，再合成一幅图。

void image_average(cv::Mat &src, cv::Mat &dst)
{
    
    vector<cv::Mat> rgb_separate;
    cv::split(src, rgb_separate);
//    imshow(rgb_separate[0]);
//    imshow(rgb_separate[1]);
//    imshow(rgb_separate[2]);
    cv::Mat B = rgb_separate[0];
    cv::Mat G = rgb_separate[1];
    cv::Mat R = rgb_separate[2];
//    cout << B << endl;
//    imshow(B);
//    imshow(G);
//    imshow(R);
    
    float sum_b;
    float sum_g;
    float sum_r;
    float b;
    float g;
    float r;
    
    for(int i = 0; i < B.rows; i++)
    {
        for(int j = 0; j < B.cols; j++)
        {
            sum_b += B.at<uchar>(i, j);
        }
    
    }
    b = sum_b/(B.cols * B.rows);

    for(int i = 0; i < B.rows; i++)
    {
        for(int j = 0; j < B.cols; j++)
        {
            
            B.at<uchar>(i, j) = b;
            
        }
        
    }
    //imshow(B);
    
    for(int i = 0; i < G.rows; i++)
    {
        for(int j = 0; j < G.cols; j++)
        {
            sum_g += G.at<uchar>(i, j);
        }
        
    }
    g = sum_g/(G.cols * G.rows);
    
    for(int i = 0; i < G.rows; i++)
    {
        for(int j = 0; j < G.cols; j++)
        {
            
            G.at<uchar>(i, j) = g;
            
        }
        
    }
    //imshow(G);
    
    for(int i = 0; i < R.rows; i++)
    {
        for(int j = 0; j < R.cols; j++)
        {
            sum_r += R.at<uchar>(i, j);
        }
        
    }
    r = sum_r/(R.cols * R.rows);
    
    for(int i = 0; i < R.rows; i++)
    {
        for(int j = 0; j < R.cols; j++)
        {
            
            R.at<uchar>(i, j) = r;
            
        }
        
    }
    //imshow(R);
    
    cout << "r:" << r <<"; " << "g:" << g <<"; " << "b:" << b << endl;
    cout << "R.rows:" << R.rows <<"; " << "R.cols:" << R.cols <<"; " << "R.rows * R.cols:" << R.rows * R.cols<<endl;
    cout << "G.rows:" << G.rows <<"; " << "G.cols:" << G.cols <<"; " << "G.rows * G.cols:" << G.rows * G.cols<<endl;
    cout << "B.rows:" << B.rows <<"; " << "B.cols:" << B.cols <<"; " << "B.rows * B.cols:" << B.rows * B.cols<<endl;
    cv::merge(rgb_separate, dst);
    
}

//------------------
//对高斯核进行归一化处理
//------------------

void normalisize(cv::Mat &kernel)
{
    double sum = 0.0;
    for(int i = 0; i < kernel.cols; i++)
    {
       
        sum += kernel.at<double>(0,i);
        //cout << "sum: " << sum << endl;
        //if (sum == 1) return;
        
    }
    //cout << "sum: " << sum << endl;
    for(int i = 0; i < kernel.cols; i++)
    {
        kernel.at<double>(0,i) = kernel.at<double>(0,i)/sum;
    }
    //cout << "norm_kernel:" << kernel << endl;
    //cout << "kernel.cols: " <<kernel.cols << endl;
}

//--------------------------
//将RGB色彩空间转化为Lab色彩空间
//--------------------------

//void RgbToLab(cv::Mat &src, cv::Mat Lab_L, cv::Mat Lab_a, cv::Mat Lab_b)
//{
//    vector<cv::Mat> rgb_separate;
//    cv::split(src, rgb_separate);
//    cv::Mat B = rgb_separate[0];
//    cv::Mat G = rgb_separate[1];
//    cv::Mat R = rgb_separate[2];
//
//    cv::Mat b;
//    cv::Mat r;
//    cv::Mat g;
//    cv::Mat g_b;
//    cv::Mat g_r;
//    cv::Mat g_g;
//    
//    
//    //    int hahaType = B.type();
//    //    std::cout<<"BType = "<<hahaType<<std::endl;
//    //    b.at<double>(i,j) = B.at<uchar>(i,j) / 255.0;
//    
//    cv::normalize(B, b, 0, 1, cv::NORM_MINMAX, CV_64F);
//    
//    //    int hahaType1 = b.type();
//    //    std::cout<<"bType = "<<hahaType1<<std::endl;
//    
//    cv::normalize(G, g, 0, 1, cv::NORM_MINMAX, CV_64F);
//    
//    cv::normalize(R, r, 0, 1, cv::NORM_MINMAX, CV_64F);
//    
//    //------------------------
//    //步骤：RGB --> XYZ
//    //------------------------
//    
//    for(int i = 0; i < b.rows; i++)
//    {
//        for(int j = 0; j < b.cols; j++)
//        {
//            if (b.at<double>(i,j) > 0.04045)
//            {
//                b.at<double>(i,j) = pow((b.at<double>(i,j) + 0.055) / 1.055, 2.4);
//            }
//            else
//                b.at<double>(i,j) = (double)(b.at<double>(i,j) / 12.92);
//            
//        }
//    }
//
//    
//    for(int i = 0; i < g.rows; i++)
//    {
//        for(int j = 0; j < g.cols; j++)
//        {
//            if (g.at<double>(i,j) > 0.04045)
//            {
//                g.at<double>(i,j) = pow((g.at<double>(i,j) + 0.055) / 1.055, 2.4);
//            }
//            else
//                g.at<double>(i,j) = (double)(g.at<double>(i,j) / 12.92);
//            
//        }
//    }
//  
//    
//    for(int i = 0; i < r.rows; i++)
//    {
//        for(int j = 0; j < r.cols; j++)
//        {
//            if (r.at<double>(i,j) > 0.04045)
//            {
//                r.at<double>(i,j) = pow((r.at<double>(i,j) + 0.055) / 1.055, 2.4);
//            }
//            else
//                r.at<double>(i,j) = (double)(r.at<double>(i,j) / 12.92);
//            
//        }
//    }
//
//    
//    //------------------------
//    //步骤：XYZ --> Lab
//    //------------------------
//
//    double scale = 1;
//    cv::Size ksize = cv::Size(b.cols * scale, b.rows * scale);
//    
//    cv::Mat X = cv::Mat(ksize,CV_64F);
//    cv::Mat Y = cv::Mat(ksize,CV_64F);
//    cv::Mat Z = cv::Mat(ksize,CV_64F);
//    Lab_L = cv::Mat(ksize,CV_64F);
//    Lab_a = cv::Mat(ksize,CV_64F);
//    Lab_b = cv::Mat(ksize,CV_64F);
//    
//    
//    
//    for(int i = 0; i < X.rows; i++)
//    {
//        for(int j = 0; j < X.cols; j++)
//        {
//            X.at<double>(i,j) = 0.4125 * r.at<double>(i,j)  + 0.3576 * g.at<double>(i,j) + 0.1805 * b.at<double>(i,j) ;
//        }
//    }
//    //cout << X << endl;
//    for(int i = 0; i < X.rows; i++)
//    {
//        for(int j = 0; j < X.cols; j++)
//        {
//            X.at<double>(i,j) = X.at<double>(i,j)/0.9506;
//            if (X.at<double>(i,j) > 0.008856)
//                X.at<double>(i,j) = pow(X.at<double>(i,j), 1.0 / 3.0);
//            else
//                X.at<double>(i,j) = (7.787 * X.at<double>(i,j)) + (16.0 / 116.0);
//        }
//    }
//    //cout << X << endl;
//    
//    for(int i = 0; i < Y.rows; i++)
//    {
//        for(int j = 0; j < Y.cols; j++)
//        {
//            Y.at<double>(i,j) =0.2126 * r.at<double>(i,j) + 0.7152 * g.at<double>(i,j)+ 0.0722 * b.at<double>(i,j) ;
//        }
//    }
//    
//    for(int i = 0; i < Y.rows; i++)
//    {
//        for(int j = 0; j < Y.cols; j++)
//        {
//            Y.at<double>(i,j) = Y.at<double>(i,j)/1.0;
//            if (Y.at<double>(i,j) > 0.008856)
//                Y.at<double>(i,j) = pow(Y.at<double>(i,j), 1.0 / 3.0);
//            else
//                Y.at<double>(i,j) = (7.787 * Y.at<double>(i,j)) + (16.0 / 116.0);
//        }
//    }
//    
//    
//    for(int i = 0; i < Z.rows; i++)
//    {
//        for(int j = 0; j < Z.cols; j++)
//        {
//            Z.at<double>(i,j) =0.0193 * r.at<double>(i,j) + 0.1192 * g.at<double>(i,j) + 0.9505 * b.at<double>(i,j) ;
//        }
//    }
//    
//    for(int i = 0; i < Z.rows; i++)
//    {
//        for(int j = 0; j < Z.cols; j++)
//        {
//            Z.at<double>(i,j) = Z.at<double>(i,j)/1.0890;
//            if (Z.at<double>(i,j) > 0.008856)
//                Z.at<double>(i,j) = pow(Z.at<double>(i,j), 1.0 / 3.0);
//            else
//                Z.at<double>(i,j) = (7.787 * Z.at<double>(i,j)) + (16.0 / 116.0);
//        }
//    }
//    
//    for(int i = 0; i < Lab_L.rows; i++)
//    {
//        for(int j = 0; j < Lab_L.cols; j++)
//        {
//            Lab_L.at<double>(i,j) = 116.0 * Y.at<double>(i,j)- 16.0;
//        }
//    }
//    
//    for(int i = 0; i < Lab_a.rows; i++)
//    {
//        for(int j = 0; j < Lab_a.cols; j++)
//        {
//            Lab_a.at<double>(i,j) =  500.0 * (X.at<double>(i,j)- Y.at<double>(i,j));
//        }
//    }
//    
//    for(int i = 0; i < Lab_b.rows; i++)
//    {
//        for(int j = 0; j < Lab_b.cols; j++)
//        {
//            Lab_b.at<double>(i,j) =  500.0 * (Y.at<double>(i,j)- Z.at<double>(i,j));
//        }
//    }
//    
//    // cout << "Lab_a: " << Lab_a << endl;
//    imshow(Lab_L);
//    imshow(Lab_a);
//    imshow(Lab_b);
//
//}

//---------------
//对图像进行高斯模糊
//---------------

void Gaussian_Blur(cv::Mat &src, cv::Mat kernel, cv::Mat &dst)
{
    vector<cv::Mat> rgb_separate;
    
    cv::split(src, rgb_separate);
    cv::Mat B = rgb_separate[0];
    //cout << "B:" << B <<endl;
    cv::Mat G = rgb_separate[1];
    cv::Mat R = rgb_separate[2];
    
    double scale = 1;
    cv::Size ksize = cv::Size(B.cols * scale, B.rows * scale);
    cv::Mat b = cv::Mat(ksize,CV_64F);
    cv::normalize(B, b, 0, 1, cv::NORM_MINMAX, CV_64F);
    
    cv::Mat g = cv::Mat(ksize,CV_64F);
    cv::normalize(G, g, 0, 1, cv::NORM_MINMAX, CV_64F);
    
    cv::Mat r = cv::Mat(ksize,CV_64F);
    cv::normalize(R, r, 0, 1, cv::NORM_MINMAX, CV_64F);
    
    
    int radius = (kernel.cols - 1)/2;
    
    //cout << "radius: " << radius << endl;
    
    for(int i = 0; i < b.rows; i++)
    {
        for(int j = 0; j < b.cols; j++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((j + k) >= 0 && (j + k) < b.cols)
                {
                    b.at<double>(i,j) = b.at<double>(i, j-2) * kernel.at<double>(0) + b.at<double>(i, j-1) * kernel.at<double>(1) + b.at<double>(i, j) * kernel.at<double>(2) + b.at<double>(i, j+1) * kernel.at<double>(3) + b.at<double>(i, j+2) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    //imshow(b);
    
    
    for(int j = 0; j < b.cols; j++)
        
    {
        for(int i = 0; i < b.rows; i++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((i + k) >= 0 && (i + k) < b.cols)
                {
                    b.at<double>(i,j) = b.at<double>(i-2, j) * kernel.at<double>(0) + b.at<double>(i-1, j) * kernel.at<double>(1) + b.at<double>(i, j) * kernel.at<double>(2) + b.at<double>(i+1, j) * kernel.at<double>(3) + b.at<double>(i+2, j) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    
//    imshow(B);
//    imshow(b);
    
    for(int i = 0; i < g.rows; i++)
    {
        for(int j = 0; j < g.cols; j++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((j + k) >= 0 && (j + k) < g.cols)
                {
                    g.at<double>(i,j) = g.at<double>(i, j-2) * kernel.at<double>(0) + g.at<double>(i, j-1) * kernel.at<double>(1) + g.at<double>(i, j) * kernel.at<double>(2) + g.at<double>(i, j+1) * kernel.at<double>(3) + g.at<double>(i, j+2) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    //imshow(b);
    
    
    for(int j = 0; j < g.cols; j++)
        
    {
        for(int i = 0; i < g.rows; i++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((i + k) >= 0 && (i + k) < b.cols)
                {
                    g.at<double>(i,j) = g.at<double>(i-2, j) * kernel.at<double>(0) + g.at<double>(i-1, j) * kernel.at<double>(1) + g.at<double>(i, j) * kernel.at<double>(2) + g.at<double>(i+1, j) * kernel.at<double>(3) + g.at<double>(i+2, j) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    
    imshow(G);
    imshow(g);
    
    for(int i = 0; i < r.rows; i++)
    {
        for(int j = 0; j < r.cols; j++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((j + k) >= 0 && (j + k) < b.cols)
                {
                    r.at<double>(i,j) = r.at<double>(i, j-2) * kernel.at<double>(0) + r.at<double>(i, j-1) * kernel.at<double>(1) + r.at<double>(i, j) * kernel.at<double>(2) + r.at<double>(i, j+1) * kernel.at<double>(3) + r.at<double>(i, j+2) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    //imshow(b);
    
    
    for(int j = 0; j < r.cols; j++)
        
    {
        for(int i = 0; i < r.rows; i++)
        {
            for(int k = - radius; k < radius; k++)
            {
                if((i + k) >= 0 && (i + k) < r.cols)
                {
                    r.at<double>(i,j) = r.at<double>(i-2, j) * kernel.at<double>(0) + r.at<double>(i-1, j) * kernel.at<double>(1) + r.at<double>(i, j) * kernel.at<double>(2) + r.at<double>(i+1, j) * kernel.at<double>(3) + r.at<double>(i+2, j) * kernel.at<double>(4);
                    //                    cout <<"i: " << i << " , " << "j:" << j << endl;
                    //                    cout << "b : " <<  b.at<double>(i,j) << endl;
                    //                    cout << "b0: " << b.at<double>(i, j-2) << " , " << " m0: " << m.at<double>(0) << " ; "
                    //                         << "b1: " << b.at<double>(i, j-1) << " , " << " m1: " << m.at<double>(1) << " ; "
                    //                         << "b2: " << b.at<double>(i, j) << " , " << " m2: " << m.at<double>(2) << " ; "
                    //                         << "b3: " << b.at<double>(i, j+1) << " , " << " m3: " << m.at<double>(3) << " ; "
                    //                         << "b4: " << b.at<double>(i, j+2) << " , " << " m4: " << m.at<double>(4) << " ; " << endl;
                    //
                }
            }
        }
    }
    
    imshow(R);
    imshow(r);
    
    vector<cv::Mat> rgb_merge;
    rgb_merge.push_back(b);
    rgb_merge.push_back(g);
    rgb_merge.push_back(r);
    
    cv::merge(rgb_merge, dst);
    //imshow(dst);

}

int main(int argc, const char * argv[])
{
    //cv::Mat src = cv::imread("/Users/mengying/Masterarbeit/Opencv-test/Opencv-test/building.jpg");
    //cv::Mat src = cv::imread("/Users/mengying/Masterarbeit/Opencv-test/Opencv-test/a.jpg");
    cv::Mat src = cv::imread("/Users/mengying/Masterarbeit/Opencv-test/Opencv-test/boldt.jpg");
    //chariot.jpg

    cv::Mat dst;
    imshow(src);
    //image_average(src, dst);
    //imshow(dst);
    //cv::Mat kernel(1,5,CV_32F);

//    cv::Mat kernel =(cv::Mat_<double>(1,5)<<1,4,6,4,1);
//    int a = kernel.at<int>(1, 2);
//    cout << "a: " << a << endl;
//    cout <<"k.cols:" << kernel.cols << endl;
//    cout << "kernel:" << kernel << endl;
//    normalisize(kernel);

    cv::Mat m(1, 5, CV_64FC1, 1);
    m.at<double>(0) = 1;
    m.at<double>(1) = 4;
    m.at<double>(2) = 6;
    m.at<double>(3) = 4;
    m.at<double>(4) = 1;
    //cout << "m: " << m << endl;
    normalisize(m);
    cout << "m: " << m << endl;
    
    cv::Mat Lab_L ;
    cv::Mat Lab_a ;
    cv::Mat Lab_b ;
    
    //RgbToLab(src, Lab_L, Lab_a , Lab_b);
    
    
    cv::Mat temp;
    Gaussian_Blur(src, m, temp);
    imshow(temp);
    //RgbToLab(temp, Lab_L, Lab_a, Lab_b);
    //cvCvtColor(&src, &temp,CV_RGB2HSV);
    //imshow(temp);
    cv::cvtColor(src, temp,CV_RGB2Lab);
    imshow(temp);
    vector<cv::Mat> Lab_separate;
    cv::split(temp, Lab_separate);
    imshow(Lab_separate[0]);
    imshow(Lab_separate[1]);
    imshow(Lab_separate[2]);
    cv::Mat temp2;
    image_average(src, temp2);
    imshow(temp2);
    vector<cv::Mat> Lab_separate2;
    cv::split(temp2, Lab_separate2);
    imshow(Lab_separate2[0]);
    imshow(Lab_separate2[1]);
    imshow(Lab_separate2[2]);
//    int type = Lab_separate[1].type();
//    cout << "Lab_separate[1] type:" << type << endl;
    Lab_separate[0].convertTo(Lab_separate[0],CV_64F,1/255.0);
    Lab_separate[1].convertTo(Lab_separate[1],CV_64F,1/255.0);
    Lab_separate[2].convertTo(Lab_separate[2],CV_64F,1/255.0);
    Lab_separate2[0].convertTo(Lab_separate2[0],CV_64F,1/255.0);
    Lab_separate2[1].convertTo(Lab_separate2[1],CV_64F,1/255.0);
    Lab_separate2[2].convertTo(Lab_separate2[2],CV_64F,1/255.0);
    
    cout << Lab_separate[0].at<double>(0,0) << endl;
    cout << Lab_separate2[0].at<double>(0, 0) << endl;
    
    
    double scale = 1;
    cv::Size ksize = cv::Size(Lab_separate[0].cols * scale, Lab_separate[0].rows * scale);
    cv::Mat S = cv::Mat(ksize,CV_64F);
    
    //cout << Lab_separate[1] << endl;
    
    //cout << Lab_separate2[2] << endl;

//    for(int i = 0 ; i < Lab_separate[0].rows; i++)
//    {
//        for(int j = 0; j < Lab_separate[0].cols; j++)
//        {
//            S.at<double>(i,j) = pow((Lab_separate[0].at<double>(i,j) - Lab_separate2[0].at<double>(i,j)),2) +  pow((Lab_separate[2].at<double>(i,j) - Lab_separate2[2].at<double>(i,j)),2);
//            
//            //cout << S.at<double>(i,j) << endl;
//        }
//    }
    
    
    
    //cout << S << endl;
    //imshow(S);


    
    

    
    
    return 0;
}

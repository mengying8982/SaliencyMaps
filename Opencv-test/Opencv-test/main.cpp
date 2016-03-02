//
//  main.cpp
//  Opencv-test
//
//  Created by 孟莹 on 16/2/10.
//  Copyright © 2016年 孟莹. All rights reserved.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include<opencv2/opencv.hpp>
#include<opencv/highgui.h>
#include<opencv/cv.h>


#define GAUS_LEVELS 9
#define Num_Of_Local_Maxima 12


#define GRAD_0 0
#define GRAD_45 1
#define GRAD_90 2
#define GRAD_135 3

using namespace std;

/* 
 funktion: imshow
 description: display an image, wait until any keypress
 calls: imshow and waitKey
 input: an image(transformed in matrix)
 output: an image named TEST
 return: void
 others: no
*/

double angle2radian(const double &a){
    return a *  CV_PI / 180.;
}

void imshow(const cv::Mat &image)
{
    cv::imshow("TEST", image);
    cv::waitKey(0);
}

/*
 funktion: createGausParymids
 description: use it to create a Gaussian parymids of image src, named m with n levels.
 calls: cv::resize
 input: 
    scr: an image(transformed in matrix);
    m: the name of the parymid;
    n: the whole level of this parymid.
 output: a parymid with n levels.
 return: void
 others: (640, 480) is the size I want to resize.
 */

//产生一个n层的高斯金字塔

void createGausParymids(const cv::Mat &src, vector<cv::Mat> &m, const int &n)
{
    cv::Mat p1;
    cv::resize(src, p1, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);
    m.push_back(p1);
    cv::Mat dst;
    for(int i = 0;i < n - 1;i++)
    {
        pyrDown(p1, dst, cv::Size(p1.cols/2, p1.rows/2));
        p1 = dst;
        m.push_back(p1);
    }
}

/*
 funktion: showGausParymids
 description: to display a Gaussian parymid and the information of each level(width and height).
 calls: cv::imshow
 input: 
    m: the name of a ceated parymid.
 output: the parymid m
 return: void
 others: no
 */

//显示所产生的高斯金字塔以及它的长宽

void showGausParymids(const vector<cv::Mat> &m)
{
    for(int i = 0; i < m.size(); i++)
    {
        cv::imshow(" ", m[i]);
        cv::waitKey(1000);
        cout << i << ": (height = " << m[i].rows << ", " << "width = " << m[i].cols << ")"<< endl;
        
    }
    
}

/*
 funktion: getLevelofGausParymids
 description: to get the image from the assigned level of parymid
 calls: no
 input: 
    m: the name of the parymid;
    level: the image of which level you want to see.
 output: the image you want
 return: no
 others: no
 */

//得到具体某一层的图片信息

cv::Mat getLevelofGausParymids(const vector<cv::Mat> &m, const int &level)
{
    int n;
    if(level > m.size())
        n = m.size();
    else if (level < 0)
        n = 0;
    else
        n = level;
    return m[n];
}

/*
 funktion: getAGaborKernel
 description: this funktion is used to create a gabor kernel
 calls: no
 input: 
    ksize: the size of the kernel; 
    sigma: standard deviation of the Gaussian envelope, sigma = 0.56*lambd;
    theta: orientation (in this programm I need 0 grad, 45 grad, 90 grad, 135 grad);
    lambd: wavelength;
    gamma: spatial aspect ratio, usually 0.5;
    psi: phase offset.
 output: a gabor kernel
 return: kernel
 others: no
 */

//得到Gabor滤波器的核

cv::Mat getAGaborKernel(cv::Size ksize, double sigma, double theta, double lambd, double gamma, double 	psi )
{
    
    int x,y;
    double xtemp, ytemp, temp1, temp2;
    double m_sigma = sigma;
    double m_theta = angle2radian(theta);
    double m_lambd = lambd;
    double m_gamma = gamma;
    double m_psi = psi;
    cv::Mat kernel(ksize.width, ksize.height, CV_64F);
  
    
    for(int i = 0; i < ksize.height; i++)
    {
        for(int j = 0; j < ksize.width; j++)
        {
            x = j - ksize.width/2;
            y = ksize.height/2 - i ;
            
            xtemp = (double)x*cos(m_theta) + (double)y*sin(m_theta);
            ytemp = (double)y*cos(m_theta) - (double)x*sin(m_theta);
            
            temp1 = exp(-(pow(xtemp, 2)+ pow(m_gamma*ytemp, 2))/(2*pow(m_sigma, 2)));
            temp2 = cos(2*CV_PI*xtemp/m_lambd + m_psi);
            
           
            
            kernel.at<double>(i, j)= temp1 * temp2; //使用.at访问Matrix中的指定数据
        }
        
    }
    return kernel;
}

/*
 funktion: acrossScaleDiff
 description: caculate the acress scale difference between two images.
 calls: cv::resize, cv::absdiff
 input: 
    scr1: input image 1;
    scr2: input image 2;
    dst: the result of across scale difference between scr1 and scr2.
 output: the result of across scale difference between scr1 and scr2
 return: void
 others: no
 */

//描述center-surround operation（across-scale difference）的具体操作; 将scr2变到和scr1一样的大小，然后进行点对点的减操作。

void acrossScaleDiff(const cv::Mat &scr1, const cv::Mat &scr2, cv::Mat &dst)
{
    double scale = 1;
    cv::Size ksize = cv::Size(scr1.cols * scale, scr1.rows * scale);
    
    cv::Mat temp = cv::Mat(ksize,CV_32F);
    cv::resize(scr2, temp, ksize, 0, 0, cv::INTER_LINEAR); //scr2通过插值法变得和scr1一样大，含有相同数目的像素值
    dst = abs(scr1 - temp);
   // cv::subtract(scr1, temp, dst); //对scr1和temp(变得和scr1一样大的scr2)进行逐点相减，取绝对值
    
}

/*
 funktion:
 description:
 calls:
 input:
 output:
 return:
 others: 
 */
//找到矩阵中第二大的值，先找到最大值，然后取零，再找一遍最大值。
double FindSecondMaxima(cv::Mat I) //值传递，不能用const，因为里面的I用了两遍，第二遍的时候改变了，不能用&，因为不能改变，只是暂时改变，只需要值，不需要改变的矩阵。
{
    double minVal, maxVal,secondMaxima;
    double *minp = &minVal;
    double *maxp = &maxVal;
    double *secondMaxp = &secondMaxima;
    cv::minMaxLoc(I, minp, maxp);
    for(int i = 0; i < I.rows; i++)
    {
        for(int j = 0; j < I.cols; j++)
        {
            if(I.at<double>(i, j) == 1)
                I.at<double>(i, j) = 0;
                
        }
    }
    cv::minMaxLoc(I, minp, secondMaxp);
    return secondMaxima;
    
}

/*
 funktion:
 description:
 calls:
 input:
 output:
 return:
 others:
 */

void N_Operation(const cv::Mat &src, cv::Mat &temp)
{
    //cv::Mat temp;
    cv::Mat I;
    vector <cv::Mat> partOfOriginalImage;
    double localMaxima = 0.0;
    double localMaxima_avg = 0.0;
   
    //cv::normalize(src, dst, 1.0, 0.0, cv::NORM_MINMAX);
    cv::normalize(src, temp, 0, 1, cv::NORM_MINMAX, CV_64F);//归一化
    //cout << "src:" <<src.rows<<", "<< src.cols<< endl;
    //cout << "temp:" <<temp.rows<<", "<< temp.cols<< endl;
    //imshow(src);
    //imshow(temp);
    
//    for(int i = 0; i <  Num_Of_Local_Maxima; i++)//Num_Of_Local_Maxima = 10
//    {
//        int temp3 = ceil(dst.cols/ Num_Of_Local_Maxima)*i;//range start, ceil: 向上取整
//        int temp4 = ceil(dst.cols/ Num_Of_Local_Maxima)*(i+1);//range end
//        I = temp(cv::Range::all(),cv::Range(temp3 ,temp4)); //取部分矩阵，每一个部分矩阵为I
//        teileVonOriginalImage.push_back(I);
////        cout << "第 " << i << "个矩阵：" << endl
////
////             << "最大值是： " << FindSecondMaxima(I) << endl;
//        localMaxima += FindSecondMaxima(I);
//      }
    
    for (int r = 0; r < temp.rows; r += temp.rows/6 )
    {
        for (int c = 0; c < temp.cols; c += temp.cols/8 )
        {
            I = temp(cv::Range(r, min(r + temp.rows/6, temp.rows)), cv::Range(c, min(c + temp.cols/8, temp.cols)));
            
            partOfOriginalImage.push_back(I);
            localMaxima += FindSecondMaxima(I);
         
            
        }
    }
    
   //cout <<"原图被分成了"<< partOfOriginalImage.size() <<"份"<< endl;
   // cout << "第一份：" <<  teileVonOriginalImage[0] << endl;
    localMaxima_avg = localMaxima/Num_Of_Local_Maxima;
    //cout << "平均值为： " << localMaxima_avg << endl;
    for(int i = 0; i < temp.rows; i++)
    {
        for (int j = 0; j < temp.cols; j++)
        {
            temp.at<double>(i,j) = temp.at<double>(i,j) * pow((1-localMaxima_avg), 2);
        }
    }
    
    
}


cv::Mat change_size(const cv::Mat &src, float scale_number) //以后放在image的namespace中
{
    if(scale_number == 0)
        return src;
    cv::Size isize = cv::Size(src.cols * scale_number, src.rows * scale_number);
    cv::Mat dst = cv::Mat(isize,CV_32F);
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
    return dst;
}

//void three_point_addition(cv::Mat &src1, cv::Mat &src2, cv::Mat &src3, cv::Mat &dst)
//{
//    
//    //dst = cv::Mat(cv::Size(src1.rows, src1.cols),CV_32F);
//
//    dst = src1 + src2 + src3;
//    //Mat m(3, 5, CV_32FC1, 1);
//}

void across_scale_additon(cv::Mat &src1, const float &scale_number1, cv::Mat &src2, const float &scale_number2, cv::Mat &src3,
                          const float &scale_number3, cv::Mat &dst)
{
    cv::Mat temp1, temp2, temp3;
    temp1 = change_size(src1, scale_number1);
//  cout <<"temp1:"<< "(" << temp1.rows << "," << temp1.cols << ")"<<endl;
//  cout << "(" << src1.rows << "," << src1.cols << ")"<<endl;
    
    temp2 = change_size(src2, scale_number2);
    
    temp3 = change_size(src3, scale_number3);
    
    dst = temp1 + temp2 + temp3;
    
//    imshow(src1);
//    imshow(temp1);
//    imshow(src2);
//    imshow(temp2);
//    imshow(src3);
//    imshow(temp3);
//    imshow(dst);
  // three_point_addition(temp1, temp2, temp3, dst);
    
}

int main(int argc, char* argv[])
{
    vector<cv::Mat> gausParymids;

   
    cv::Mat src = cv::imread("/Users/mengying/Masterarbeit/Opencv-test/Opencv-test/building.jpg");
    imshow(src);
    
    createGausParymids(src, gausParymids, GAUS_LEVELS);
//  showGausParymids(gausParymids);
//    cout << "src: ("<< src.rows << "," << src.cols << ")"<< endl;
//    cout << "gausParymids: ("<< gausParymids[0].rows << "," << gausParymids[0].cols << ")"<< endl;

    
    //分离图像的I,R,G,B 特征值信息
    vector<cv::Mat> rgb_separate;
//  vector<vector<cv::Mat>> gausparymid_features;
    vector<cv::Mat> intensity_parymid;
    vector<cv::Mat> red_parymid;
    vector<cv::Mat> green_parymid;
    vector<cv::Mat> blue_parymid;
    vector<cv::Mat> yellow_parymid;
   

   
    cv::Mat I, R, G, B, Y;

    
    
    for(int i = 0; i < GAUS_LEVELS; i++)
    {
        cv::split(getLevelofGausParymids(gausParymids, i), rgb_separate);//split each feachure of the image,r:2,g:1,b:0.

        rgb_separate[0].convertTo(rgb_separate[0], CV_32FC1, 1.0/255.0);
        rgb_separate[1].convertTo(rgb_separate[1], CV_32FC1, 1.0/255.0);
        rgb_separate[2].convertTo(rgb_separate[2], CV_32FC1, 1.0/255.0);
        
        I = (rgb_separate[0] + rgb_separate[1] + rgb_separate[2])/3.0;
        for(int j = 0; j < I.rows; j++)
        {
            for (int k = 0; k < I.cols; k++)
            {
                if(I.at<float>(j,k) < 0 )
                    I.at<float>(j,k) = 0;
            }
        }

        
        R = rgb_separate[2] - (rgb_separate[1] + rgb_separate[0])/2.0;
        for(int j = 0; j < R.rows; j++)
        {
            for (int k = 0; k < R.cols; k++)
            {
                if(R.at<float>(j,k) < 0 )
                    R.at<float>(j,k) = 0;
            }
        }
        
        G = rgb_separate[1] - (rgb_separate[2] + rgb_separate[0])/2.0;
        for(int j = 0; j < G.rows; j++)
        {
            for (int k = 0; k < G.cols; k++)
            {
                if(G.at<float>(j,k) < 0 )
                    G.at<float>(j,k) = 0;
            }
        }
        
        B = rgb_separate[0] - (rgb_separate[1] + rgb_separate[2])/2.0;
        for(int j = 0; j < B.rows; j++)
        {
            for (int k = 0; k < B.cols; k++)
            {
                if(B.at<float>(j,k) < 0 )
                    B.at<float>(j,k) = 0;
            }
        }
        
        Y = (rgb_separate[1] + rgb_separate[2])/2.0 - abs(rgb_separate[1] - rgb_separate[2])/2.0 - rgb_separate[0];
        for(int j = 0; j < Y.rows; j++)
        {
            for (int k = 0; k < Y.cols; k++)
            {
                if(Y.at<float>(j,k) < 0 )
                    Y.at<float>(j,k) = 0;
            }
        }
        
//        if (i == 8)
//        {
//            cout << "I:" << I << endl;
//            cout << "R:" << R <<endl;
//            cout << "G:" << G << endl;
//            cout << "B:" << B << endl;
//            cout << "Y:" << Y << endl;
//        }

        intensity_parymid.push_back(I);
        red_parymid.push_back(R);
        green_parymid.push_back(G);
        blue_parymid.push_back(B);
        yellow_parymid.push_back(Y);
        
        rgb_separate.clear();
        
    }
//    showGausParymids(intensity_parymid);
//    showGausParymids(red_parymid);
//    showGausParymids(green_parymid);
//    showGausParymids(blue_parymid);
//    showGausParymids(yellow_parymid);

  
//    gausparymid_features.push_back(intensity_parymid);
//    gausparymid_features.push_back(red_parymid);
//    gausparymid_features.push_back(green_parymid);
//    gausparymid_features.push_back(blue_parymid);
//    gausparymid_features.push_back(yellow_parymid);
//
//    
//    //cv::imshow("a",getLevelofGausParymids(intensity_parymid,1));
//    
//    //cout << gausparymid_features.size() << "   " << gaus_features[0].size() << endl;
//    
    vector<cv::Mat> grad_parymids[4]; //grad_0_parymid, grad_45_parymid, grad_90_parymid, grad_135_parymid
    
    cv::Mat dst;
    
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < GAUS_LEVELS; j++)
        {
            //cv::filter2D(getLevelofGausParymids(intensity_parymid,i), dst, getLevelofGausParymids(intensity_parymid,i).depth(), getAGaborKernel(cv::Size(100,100), 100, 45*j, 10, 1.0, 0));
            
            cv::filter2D(getLevelofGausParymids(intensity_parymid,j), dst, getLevelofGausParymids(intensity_parymid,j).depth(), getAGaborKernel(cv::Size(10,10), 100, 45*i, 10, 1.0, 0));
            //cv::Size ksize, double sigma, double theta, double lambd, double gamma, double 	psi
            //cv::filter2D(getLevelofGausParymids(intensity_parymid,j), dst, -1, getAGaborKernel(cv::Size(31,31), 1, 45*i, 1.0, 0.02, 0));
            
            
            grad_parymids[i].push_back(dst);
           
        }
    
    }
//    imshow(getLevelofGausParymids(intensity_parymid, 2));
//    
//    imshow(getLevelofGausParymids(intensity_parymid, 5));
//    acrossScaleDiff(getLevelofGausParymids(intensity_parymid, 2), getLevelofGausParymids(intensity_parymid, 5), dst);
//    
//    imshow(dst);// I(2,5)
 

//    showGausParymids(grad_parymids[0]);
    
//    到目前为止共产生了4个color金字塔intensity_parymid，red_parymid，green_parymid，blue_parymid，yellow_parymid。1个Intensity金字塔intensity_parymid。4个方  向金字塔grad_parymids[4]（由数组组成）。


//    imshow(getAGaborKernel(cv::Size(10,10), 100, 45 *i, 10, 1.0, 0));
    
//    imshow(getAGaborKernel(cv::Size(100,100), 5.6, 45*i, 10, 0.5, 0));
//    showGausParymids(grad_parymids[0]);
    
//    imshow(getAGaborKernel(cv::Size(10,10), 100, 45 *i, 10, 1.0, 0));
   
//    imshow(getLevelofGausParymids(grad_parymids[0],0));//grad 0, scale 0
//    imshow(getLevelofGausParymids(grad_parymids[1],0));
//    imshow(getLevelofGausParymids(grad_parymids[2],0));
//    imshow(getLevelofGausParymids(grad_parymids[3],0));

//    for(int i = 0;i < 4;i++)
//        gausparymid_features.push_back(grad_parymids[i]);
    
    vector<cv::Mat> acrossScaleDiff_I;
    vector<cv::Mat> acrossScaleDiff_I_2;
    vector<cv::Mat> acrossScaleDiff_RG;
    vector<cv::Mat> acrossScaleDiff_RG_2;
    vector<cv::Mat> acrossScaleDiff_BY;
    vector<cv::Mat> acrossScaleDiff_BY_2;
    vector<cv::Mat> acrossScaleDiff_O;
    vector<cv::Mat> acrossScaleDiff_O_2;
    vector<cv::Mat> acrossScaleDiff_Features;
    cv::Mat temp1, temp2, temp3, temp4, temp_acrossScaleDiff, temp_acrossScaleDiff1, temp_acrossScaleDiff2, temp_acrossScaleDiff3, temp_acrossScaleDiff4, temp_acrossScaleDiff5, temp_acrossScaleDiff6, temp_acrossScaleDiff7;
    
    for(int c = 2; c < 5 ; c++)
    {
        temp1 = getLevelofGausParymids(red_parymid,c) - getLevelofGausParymids(green_parymid,c);
        temp3 = getLevelofGausParymids(blue_parymid,c) - getLevelofGausParymids(yellow_parymid,c);
        
        //int delta = 3;
        int s = c + 3;
        
        acrossScaleDiff(getLevelofGausParymids(intensity_parymid,c), getLevelofGausParymids(intensity_parymid,s), temp_acrossScaleDiff);
        acrossScaleDiff_I.push_back(temp_acrossScaleDiff);
          
        temp2 =getLevelofGausParymids(green_parymid,s) - getLevelofGausParymids(red_parymid,s);
        acrossScaleDiff(temp1, temp2, temp_acrossScaleDiff1);
        acrossScaleDiff_RG.push_back(temp_acrossScaleDiff1);
            
        temp4 = getLevelofGausParymids(yellow_parymid,s) - getLevelofGausParymids(blue_parymid,s);
        acrossScaleDiff(temp3, temp4, temp_acrossScaleDiff2);
        acrossScaleDiff_BY.push_back(temp_acrossScaleDiff2);
            
        for(int i = 0; i < 4; i++)
        {
            acrossScaleDiff(getLevelofGausParymids(grad_parymids[i],c), getLevelofGausParymids(grad_parymids[i],s), temp_acrossScaleDiff3);
            acrossScaleDiff_O.push_back(temp_acrossScaleDiff3);
        }
            
   
        
    }
//    imshow(acrossScaleDiff_I[0]);
//    imshow(acrossScaleDiff_I[1]);
//    imshow(acrossScaleDiff_I[2]);
//    imshow(acrossScaleDiff_RG[0]);
//    imshow(acrossScaleDiff_RG[1]);
//    imshow(acrossScaleDiff_RG[2]);
//    imshow(acrossScaleDiff_BY[0]);
//    imshow(acrossScaleDiff_BY[1]);
//    imshow(acrossScaleDiff_BY[2]);

    
    for(int c = 2; c < 5 ; c++)
    {
        temp1 = getLevelofGausParymids(red_parymid,c) - getLevelofGausParymids(green_parymid,c);
        temp3 = getLevelofGausParymids(blue_parymid,c) - getLevelofGausParymids(yellow_parymid,c);

        //int delta_2 = 4;
        int s = c + 4;
    
        acrossScaleDiff(getLevelofGausParymids(intensity_parymid,c), getLevelofGausParymids(intensity_parymid,s), temp_acrossScaleDiff4);
        acrossScaleDiff_I_2.push_back(temp_acrossScaleDiff4);
            
        temp2 =getLevelofGausParymids(green_parymid,s) - getLevelofGausParymids(red_parymid,s);
        acrossScaleDiff(temp1, temp2, temp_acrossScaleDiff5);
        acrossScaleDiff_RG_2.push_back(temp_acrossScaleDiff5);
            
        temp4 = getLevelofGausParymids(yellow_parymid,s) - getLevelofGausParymids(blue_parymid,s);
        acrossScaleDiff(temp3, temp4, temp_acrossScaleDiff6);
        acrossScaleDiff_BY_2.push_back(temp_acrossScaleDiff6);
            
        for(int i = 0; i < 4; i++)
        {
            acrossScaleDiff(getLevelofGausParymids(grad_parymids[i],c), getLevelofGausParymids(grad_parymids[i],s), temp_acrossScaleDiff7);
            acrossScaleDiff_O_2.push_back(temp_acrossScaleDiff7);
        }
            
            
    }
//    cout << "acrossScaleDiff_I.size:" << acrossScaleDiff_I.size() << endl;
//    cout << "acrossScaleDiff_I_2.size:"<< acrossScaleDiff_I.size() << endl;

//    imshow(acrossScaleDiff_I_2[0]);
//    imshow(acrossScaleDiff_I_2[1]);
//    imshow(acrossScaleDiff_I_2[2]);
//    imshow(acrossScaleDiff_RG_2[0]);
//    imshow(acrossScaleDiff_RG_2[1]);
//    imshow(acrossScaleDiff_RG_2[2]);
//    imshow(acrossScaleDiff_BY_2[0]);
//    imshow(acrossScaleDiff_BY_2[1]);
//    imshow(acrossScaleDiff_BY_2[2]);
    
//    cv::Mat a;
//    
//    N_Operation(acrossScaleDiff_I_2[0], a);

    

//    imshow(acrossScaleDiff_O[0]);// 0表示acrossScaleDiff循环的第一个,c=2, 所以和第三层的大小一样



////到目前为止Intensity: 6幅图， Color: 12幅图， Orientations: 24幅图。
    
//对所有的map归一化处理：将所有的像素值转化在[0,1]之间
    vector<cv::Mat> normalize_acrossScaleDiff_I;
    vector<cv::Mat> normalize_acrossScaleDiff_RG;
    vector<cv::Mat> normalize_acrossScaleDiff_BY;
    vector<cv::Mat> normalize_acrossScaleDiff_O;
    vector<cv::Mat> tempVector;
    vector<vector<cv::Mat>>normalize_arossScaleDiff_O_Grad;
    cv::Mat temp5, temp6, temp7, temp8;
    
    
    for(int j = 0; j < 3; j++)
    {
        N_Operation(acrossScaleDiff_I[j], temp5);
        normalize_acrossScaleDiff_I.push_back(temp5);
        N_Operation(acrossScaleDiff_RG[j], temp6);
        normalize_acrossScaleDiff_RG.push_back(temp6);
        N_Operation(acrossScaleDiff_BY[j], temp7);
        normalize_acrossScaleDiff_BY.push_back(temp7);
    }
//  根据角度分成了四个Vector，每个Vector有三个元素。
    for (int  i = 0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            N_Operation(acrossScaleDiff_O[i+j*4], temp8);
            tempVector.push_back(temp8);
            
        }
      normalize_arossScaleDiff_O_Grad.push_back(tempVector);
      tempVector.clear();
    }
    //cout << "normalize_acrossScaleDiff_I.size: " << normalize_acrossScaleDiff_I.size() << endl;
    //cout <<normalize_acrossScaleDiff_I[1]<<endl;
//    imshow(acrossScaleDiff_I[0]);
//    imshow(normalize_acrossScaleDiff_I[0]);
//    imshow(acrossScaleDiff_I[1]);
//    imshow(normalize_acrossScaleDiff_I[1]);
//    imshow(acrossScaleDiff_I[2]);
//    imshow(normalize_acrossScaleDiff_I[2]);
    


    
    cv::Mat temp9, temp10, temp11, temp12, temp_0, temp_45, temp_90, temp_135;
    cv::Mat I_mean,  N_I_mean;
    across_scale_additon(normalize_acrossScaleDiff_I[0], 1.0/4.0, normalize_acrossScaleDiff_I[1], 1.0/2.0, normalize_acrossScaleDiff_I[2], 1.0, I_mean);
    
    N_Operation(I_mean, N_I_mean);
//    imshow(N_I_mean);
//
//    
    cv::Mat dst1, dst2, dst3, dst4;
    cv::Mat C_mean, N_C_mean;
    temp10 = normalize_acrossScaleDiff_RG[0] + normalize_acrossScaleDiff_BY[0];
    temp11 = normalize_acrossScaleDiff_RG[1] + normalize_acrossScaleDiff_BY[1];
    temp12 = normalize_acrossScaleDiff_RG[2] + normalize_acrossScaleDiff_BY[2];
    across_scale_additon(temp10, 1.0/4.0, temp11, 1.0/2.0, temp12, 1.0, C_mean);
    
    
    
    //cout << "(" << dst1.rows << "," << dst1.cols << ")";
    N_Operation(C_mean, N_C_mean);
   
    
    
    across_scale_additon(normalize_arossScaleDiff_O_Grad[0][0], 1.0/4.0,
                         normalize_arossScaleDiff_O_Grad[0][1], 1.0/2.0,
                         normalize_arossScaleDiff_O_Grad[0][2], 1.0,
                         temp_0);
    N_Operation(temp_0, dst2);


    across_scale_additon(normalize_arossScaleDiff_O_Grad[1][0], 1.0/4.0,
                         normalize_arossScaleDiff_O_Grad[1][1], 1.0/2.0,
                         normalize_arossScaleDiff_O_Grad[1][2], 1.0,
                         temp_45);
    N_Operation(temp_45, dst3);
    
    across_scale_additon(normalize_arossScaleDiff_O_Grad[2][0], 1.0/4.0,
                         normalize_arossScaleDiff_O_Grad[2][1], 1.0/2.0,
                         normalize_arossScaleDiff_O_Grad[2][2], 1.0,
                         temp_90);
    N_Operation(temp_90, dst4);
    
    across_scale_additon(normalize_arossScaleDiff_O_Grad[3][0], 1.0/4.0,
                         normalize_arossScaleDiff_O_Grad[3][1], 1.0/2.0,
                         normalize_arossScaleDiff_O_Grad[3][2], 1.0,
                         temp_90);
    N_Operation(temp_90, dst4);
    
    cv::Mat N_O_mean;
    cv::Mat O_mean = dst2 + dst3 + dst4;
    N_Operation(O_mean, N_O_mean);
   

    cv::Mat S = (N_I_mean +  N_C_mean +  N_O_mean)/3.0;
    imshow(getLevelofGausParymids(gausParymids, 0));
    cv::Mat N_I_mean_, N_C_mean_, N_O_mean_;

    cv::resize(N_I_mean, N_I_mean_, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);
    cv::resize(N_C_mean, N_C_mean_, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);
    cv::resize(N_O_mean, N_O_mean_, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);

//    imshow(N_C_mean_);
//    imshow(N_I_mean_);
//    imshow(N_O_mean_);
    imshow(S);
    cv::Mat S_;

    cv::resize(S, S_, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);
    imshow(S_);
//
//    
//
//    
//    
//    
//    
//    

    
    
    return 0;
}

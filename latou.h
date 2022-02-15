#ifndef LATOU_H
#define LATOU_H
#include <memory>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <sys/stat.h>
#include "opencv2/core.hpp"
using namespace std;
using namespace cv;
struct ROISAB
{
    int x1;   //左上角坐标
    int y1;   
    int x2;   //右下角坐标
    int y2;
    ROISAB():x1(0),y1(0),x2(0),y2(0){};
    Rect tange(){ return Rect(x1,y1,x2-x1,y2-y1);};
    void p(){
        cout<<x1<<","<<y1<<","<<x2<<","<<y2<<endl;
    }
};

class LaTou
{
public:
    LaTou();
    void set(string imagefolder,string classId);
    void writeDetector();
    void loadDetector();
    void sub(Mat background_,Mat object_,Mat& diff);  //差分(前景与背景进行差分,提取到前景物体)
    bool segDizuo(Mat object_,Mat background_);       //利用拉头进行是否空物体判定操作
    bool train(Mat templ,Mat background);             //形状匹配训练函数
    bool secondIdentify(Mat templ,Mat test,Point2f center,float angle);   //目标进行差分,判断是否缺乏,roi相对于模板图像旋转中心的坐标
    int predict(Mat test_img,Mat templ_image,Mat background);    //blank:1,ok:2,false:3
    void seg(Mat object_,Mat background_,int blank_thres,string blank,string file);
    void getSegRegion(Mat background_);
    int m_isSec;           //是否进行二次判断
    string m_imagefolder;  //存放的文件夹
    string m_classId;      //类型标识
    int m_numFeatures;     //特征数目
    int m_weakThreshold;   //低阈值
    int m_strongThreshold; //高阈值
    int m_angleStart;      //起始角度
    int m_angleEnd;        //终止角度
    ROISAB m_train;        //训练图片位置
    ROISAB m_logo;         //logo的位置
    int m_blankThres;      //底座空白判断阈值
    int m_firstThres;      //一次判断阈值
    int m_secondThres;     //二次判断阈值
    bool m_isLoad;         //是否已经添加模型
    Mat m_segRegion;       //判断是否为空图区域
    Mat m_back_gray;       //判断指定区域空图
    int m_count;           //判断是否为空图像素点数
};

//接口
class Interface
{
public:
    //train：1拉片和底座都训练成功 2底座训练失败 3拉片训练失败 4拉片和底座都训练失败
    int train(string imagefolder,int is_sec,ROISAB dizuo_train,ROISAB dizuo_logo,ROISAB lapian_train,ROISAB lapian_logo);
    //blank:1,ok:2,false:3
    int  predict(string imagefolder);      //预测
    static void createLog(const string path);
    void draw(string imagefolder,int is_sec,ROISAB dizuo_train,ROISAB dizuo_logo,ROISAB lapian_train,ROISAB lapian_logo);  //画图
    void testpicture(string imagefolder,string dizuopath,string lapianpath);  //批量测试图片
private:
    LaTou m_dizuo;                           //底座类
    LaTou m_lapian;                          //拉片类
};

#endif



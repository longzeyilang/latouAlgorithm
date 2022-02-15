#include "latou.h"

// /media/gzy/f093187a-674a-48c6-8baf-b04e5e3a9a0a/latou/latouAlgorithm/temporal_latou/ 1 1 50 26 326 282 208 78 258 234 470 134 792 384 464 206 522 304

int main(int argc,char **argv)
{
    string imagefolder=argv[1];                     //图片路径 "/home/gzy/latou/images/temporal_latou/" 
    int    is_train=atoi(argv[2]);                  //是否进行训练   0：不训练 1：训练
    int    is_sec=atoi(argv[3]);                    //是否进行二次判断 0:不判断 1:判断
    ROISAB dizuo_train,dizuo_logo,lapian_train,lapian_logo;
    dizuo_train.x1=atoi(argv[4]);                   //底座训练区域
    dizuo_train.y1=atoi(argv[5]);
    dizuo_train.x2=atoi(argv[6]);
    dizuo_train.y2=atoi(argv[7]);
    dizuo_logo.x1=atoi(argv[8]);                    //底座logo区域
    dizuo_logo.y1=atoi(argv[9]);
    dizuo_logo.x2=atoi(argv[10]);
    dizuo_logo.y2=atoi(argv[11]);

    lapian_train.x1=atoi(argv[12]);                 //拉片训练区域
    lapian_train.y1=atoi(argv[13]);
    lapian_train.x2=atoi(argv[14]);
    lapian_train.y2=atoi(argv[15]);
    lapian_logo.x1=atoi(argv[16]);                  //拉片logo区域
    lapian_logo.y1=atoi(argv[17]);
    lapian_logo.x2=atoi(argv[18]);
    lapian_logo.y2=atoi(argv[19]);

    
    // createLog(imagefolder);
    // if(is_train==0||is_train==1)
    //     interface(imagefolder,is_train,is_sec,dizuo_train,dizuo_logo,lapian_train,lapian_logo);
    // else if(is_train==2)
    //     draw(imagefolder,is_sec,dizuo_train,dizuo_logo,lapian_train,lapian_logo);
    // else if(is_train==3)
    // {
    //     //进行批量测试
    //     string dizuopath=argv[20];
    //     string lapianpath=argv[21];
    //     testpicture(imagefolder,dizuopath,lapianpath);
    // }
    Interface::createLog(imagefolder);  //日志文件
    Interface interface;
    // if(is_train)
    //     interface.train(imagefolder,is_sec,dizuo_train,dizuo_logo,lapian_train,lapian_logo);
    // if(is_train==0)
    // {
    //     int a=0;
    //     while(a++<10)
    //     {
    //         interface.predict(imagefolder);
    //     }
    // } 

    interface.train(imagefolder,is_sec,dizuo_train,dizuo_logo,lapian_train,lapian_logo);  //判断是否训练成功
    int a=0;
    while(a++<10)
    {
        std::cout<<interface.predict(imagefolder)<<std::endl;      //测试
    }
    return 0;
}
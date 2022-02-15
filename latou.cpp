#include "latou.h"
#include "shapeMatch.h"
#include "Log.h"
void LaTou::writeDetector()
{
    string path=m_imagefolder+m_classId+"detector.yaml";
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs<<"num_features"<<m_numFeatures;
    fs<<"weak_threshold"<<m_weakThreshold;
    fs<<"strong_threshold"<<m_strongThreshold;
    fs<<"angle_start"<<m_angleStart;
    fs<<"angle_end"<<m_angleEnd;
    fs<<"blank_thres"<<m_blankThres;
    fs<<"first_thres"<<m_firstThres;
    fs<<"second_thres"<<m_secondThres;
    fs<<"isSec"<<m_isSec;
    fs<<"m_train_x1"<<m_train.x1;
    fs<<"m_train_y1"<<m_train.y1;
    fs<<"m_train_x2"<<m_train.x2;
    fs<<"m_train_y2"<<m_train.y2;
    fs<<"m_logo_x1"<<m_logo.x1;
    fs<<"m_logo_y1"<<m_logo.y1;
    fs<<"m_logo_x2"<<m_logo.x2;
    fs<<"m_logo_y2"<<m_logo.y2;
    fs.release();
}

void LaTou::loadDetector()
{
    string path=m_imagefolder+m_classId+"detector.yaml";
    cv::FileStorage fs(path,cv::FileStorage::READ);
    fs["num_features"]>>m_numFeatures;
    fs["weak_threshold"]>>m_weakThreshold;
    fs["strong_threshold"]>>m_strongThreshold;
    fs["angle_start"]>>m_angleStart;
    fs["angle_end"]>>m_angleEnd;
    fs["blank_thres"]>>m_blankThres;
    fs["first_thres"]>>m_firstThres;
    fs["second_thres"]>>m_secondThres;
    fs["isSec"]>>m_isSec;
    fs["m_train_x1"]>>m_train.x1;
    fs["m_train_y1"]>>m_train.y1;
    fs["m_train_x2"]>>m_train.x2;
    fs["m_train_y2"]>>m_train.y2;
    fs["m_logo_x1"]>>m_logo.x1;
    fs["m_logo_y1"]>>m_logo.y1;
    fs["m_logo_x2"]>>m_logo.x2;
    fs["m_logo_y2"]>>m_logo.y2;
    fs.release();
}

LaTou::LaTou()
{
    m_isLoad=false;
}

void LaTou::set(string imagefolder,string classId)
{
    m_imagefolder=imagefolder;
    m_classId=classId;
}

void LaTou::sub(Mat background_,Mat object_,Mat& diff)
{
    Mat background,object;
    background_.convertTo(background,CV_32F);
    object_.convertTo(object,CV_32F);
    subtract(object,background,diff); //差分后图像
    convertScaleAbs(diff,diff);       //转化为绝对值
    diff.convertTo(diff,CV_8U);       //数据类型转换   
}

void LaTou::getSegRegion(Mat background_)
{
    background_=background_(m_train.tange());
    Mat back_gray,back_gray_thres;
    Mat back=background_.clone();
    if(back.channels()==3) cvtColor(back,back_gray,CV_BGR2GRAY);         //背景转化维灰度图

    //寻找中心点背景图: 图片分为中间有孔和中间无孔
    threshold(back_gray,back_gray_thres,m_blankThres,255,THRESH_OTSU);
    back_gray_thres=255-back_gray_thres;
    vector<vector<Point>>contours;
    vector<Vec4i> hierarchy;
    findContours(back_gray_thres,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    double max_area=back_gray_thres.rows*back_gray_thres.cols*0.2;
    int index=-1;
    for(int i=0;i<contours.size();i++)
        if(contourArea(contours[i])>max_area)
        {
            max_area=contourArea(contours[i]);
            index=i;
        }
    std::cout<<"index:"<<index<<std::endl;
    //没有存在最大中心位置，全局来计算
    m_segRegion=Mat(back_gray.size(),CV_8UC1,{1});
    m_count=back_gray_thres.rows*back_gray_thres.cols*0.2;
    //如果存在最大中心位置
    if(index!=-1)
    {
          //外接矩形和去掉中心图
        Rect boundRect=boundingRect(contours[index]);
        int top=std::max(boundRect.tl().y,0);
        int bottom=std::min(boundRect.br().y,m_segRegion.rows-1);
        for(int i=top;i<=bottom;i++)
            memset(m_segRegion.ptr<uchar>(i),0,m_segRegion.cols);
        m_count=(back_gray_thres.rows-(bottom-top))*back_gray_thres.cols*0.2;
    }
    //相乘
    multiply(back_gray,m_segRegion,m_back_gray);
}



bool LaTou::segDizuo(Mat object_,Mat background_)
{
    //进行裁剪
    object_=object_(m_train.tange());
    Mat object_gray,diff;
    if(object_.channels()==3) cvtColor(object_,object_gray,CV_BGR2GRAY); //目标物体图转化为灰度图

    //寻找中心点背景图: 图片分为中间有孔和中间无孔
    // threshold(back_gray,back_gray_thres,m_blankThres,255,THRESH_OTSU);
    // back_gray_thres=255-back_gray_thres;
    // vector<vector<Point>>contours;
    // vector<Vec4i> hierarchy;
    // findContours(back_gray_thres,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    // double max_area=0;
    // Point mc;
    // int index=-1;
    // for(int i=0;i<contours.size();i++)
    //     if(contourArea(contours[i])>max_area)
    //     {
    //         max_area=contourArea(contours[i]);
    //         index=i;
    //     }
    
    // //外接矩形和去掉中心图
    // Rect boundRect=boundingRect(contours[index]);
    // Mat region=Mat(back_gray.size(),CV_8UC1,{1});
    // int top=std::max(boundRect.tl().y,0);
    // int bottom=std::min(boundRect.br().y,region.rows-1);
    // for(int i=top;i<=bottom;i++)
    //     memset(region.ptr<uchar>(i),0,region.cols);

    //相乘去除中心区域
    //multiply(back_gray,region,back_gray);
    multiply(object_gray,m_segRegion,object_gray);
    sub(m_back_gray,object_gray,diff);

    //设置像素计数大小
    Mat graythres;
    cv::threshold(diff,graythres,m_blankThres,255,THRESH_BINARY); 
    cv::imwrite(m_imagefolder+"blank.jpg",graythres);            //保存图像
    int curr=countNonZero(graythres);
    Log::Info("dizuo blank thresh,count thresh and curr thresh:"+to_string(m_blankThres)+","+to_string(m_count)+","+to_string(curr));
    if(curr>m_count) return true;
    else return false;
}

bool LaTou::train(Mat templ,Mat background)
{
    //删除文件
    remove(string(m_imagefolder+m_classId+".yaml").c_str());
    remove(string(m_imagefolder+m_classId+"info.yaml").c_str());

    //ROI进行裁剪
    templ=templ(m_train.tange());
    background=background(m_train.tange());

     //对拉片和底座进行特征提取范围限制,mask进行修改
    Mat mask=Mat(templ.size(),CV_8UC1,{255});
    Mat templ_gray,background_gray,diff,thresh_background;
    cvtColor(background,background_gray,COLOR_BGR2GRAY);
    cvtColor(templ,templ_gray,COLOR_BGR2GRAY);
    sub(background_gray,templ_gray,diff);
    threshold(diff,thresh_background,230,255,THRESH_OTSU); //最大类间法
    Mat element=getStructuringElement(MORPH_RECT,Size(3,3));
    dilate(thresh_background,thresh_background,element,Point(-1,1),3,BORDER_CONSTANT);
    imwrite(string(m_imagefolder+m_classId+"_template_thresh.jpg"),thresh_background);
    multiply(mask,thresh_background,mask);


    int padding=50;
    cv::Mat padding_img=cv::Mat(templ.rows+2*padding,templ.cols+2*padding,templ.type(),cv::Scalar::all(0));
    templ.copyTo(padding_img(Rect(padding,padding,templ.cols,templ.rows)));
    cv::Mat padding_mask=cv::Mat(mask.rows+2*padding,templ.cols+2*padding,mask.type(),cv::Scalar::all(0));
    mask.copyTo(padding_mask(Rect(padding,padding,templ.cols,templ.rows)));
    
    //形状初始化和检测头
    Detector detector(m_numFeatures,{4,8},m_weakThreshold,m_strongThreshold);
    shapeInfo_producer shapes;
    shapes.set(padding_img, padding_mask);
    shapes.angle_range ={(float)m_angleStart,(float)m_angleEnd};
    shapes.angle_step=1;
    shapes.produce_infos(); //产生旋转和缩放信息
    vector<shapeInfo_producer::Info> infos_have_templ;
    for(auto& info: shapes.infos)
    {   
        //角度大于0,往逆时针方向旋转
        int templ_id = detector.addTemplate(shapes.src_of(info), m_classId, shapes.mask_of(info));
        if(templ_id==-1) 
        {
            cout<<"fail train:"<<m_classId<<endl;
            Log::Error("fail train "+m_classId);
            return false;
        }
        else infos_have_templ.push_back(info);
    }
    //创建新的文件
    detector.writeClasses(m_imagefolder+"%s.yaml");
    shapes.save_infos(infos_have_templ,m_imagefolder+m_classId+"info.yaml");
    Log::Info("save successfully "+m_classId+".yaml "+m_classId+"info.yaml ");
    padding_img.release();
    padding_mask.release();
    return true;
};

bool LaTou::secondIdentify(Mat templ,Mat test,Point2f center,float angle)
{
    //目标图像进行反旋转
    Mat M=getRotationMatrix2D(center,-angle,1.0);  
    Mat padding_dst;
    warpAffine(test,padding_dst,M,test.size());
    
    //模板图像进行抠图
    Rect logoroi=m_logo.tange();
    Rect trainroi=m_train.tange();
    Mat trainImage=templ(trainroi);  //训练图像
    Mat templ_roi=templ(logoroi);    //logo图像
    //左上角点进行减
    float x_=logoroi.x-trainroi.x-trainImage.cols/2.;   
    float y_=logoroi.y-trainroi.y-trainImage.rows/2.;
    //测试图像进行抠图
    Point2f lt=center+Point2f(x_,y_);
    Rect roi1(int(lt.x),int(lt.y),logoroi.width,logoroi.height);
    // //进行画图
    // Mat padding=padding_dst.clone();
    // rectangle(padding,roi1,CV_RGB(255,0,0)); 
    // imshow("roi",padding);
    // waitKey(0);
    // destroyWindow("roi");

    //区域位置判断
    if(roi1.tl().x<0||roi1.tl().y<0||roi1.br().x>=padding_dst.cols||roi1.br().y>=padding_dst.rows)
        return false;
    padding_dst=padding_dst(roi1).clone();
    //进行梯度计算后,再投影比对
    if(templ_roi.channels()==3) cvtColor(templ_roi,templ_roi,CV_BGR2GRAY);
    if(padding_dst.channels()==3) cvtColor(padding_dst,padding_dst,CV_BGR2GRAY);
    //进行模糊下
    GaussianBlur(templ_roi,templ_roi,Size(7,7),0);
    GaussianBlur(padding_dst,padding_dst,Size(7,7),0);

    //求取X方向的梯度
    Mat srcSobelX,srcSobelY;
    Sobel(templ_roi,srcSobelX,CV_32F,1,0,3,1.0,0);
    Sobel(templ_roi,srcSobelY,CV_32F,0,1,3,1.0,0);
    //求取Y方向的梯度
    Mat dstSobelX,dstSobelY;
    Sobel(padding_dst,dstSobelX,CV_32F,1,0,3,1.0,0);
    Sobel(padding_dst,dstSobelY,CV_32F,0,1,3,1.0,0);    

    int cols=templ_roi.cols;
    int rows=templ_roi.rows;
    cv::AutoBuffer<int> project;
    project.allocate(4*rows);      //分配内存空间
    for(size_t i = 0; i < rows; i++)
    {
        float* src_x=srcSobelX.ptr<float>(i);
        float* src_y=srcSobelY.ptr<float>(i);
        float* dst_x=dstSobelX.ptr<float>(i);
        float* dst_y=dstSobelY.ptr<float>(i);
         
        int sum_src_x=0,sum_src_y=0,sum_dst_x=0,sum_dst_y=0;
        for(size_t j = 0; j < cols; j++)
        {
            sum_src_x+=cv::abs(src_x[j]);
            sum_src_y+=cv::abs(src_y[j]); 
            sum_dst_x+=cv::abs(dst_x[j]); 
            sum_dst_y+=cv::abs(dst_y[j]); 
        }
        //取整
        project[i]=cvRound(sum_src_x/cols);      //每行的梯度的均值作为特征
        project[rows+i]=cvRound(sum_src_y/cols);
        project[2*rows+i]=cvRound(sum_dst_x/cols);
        project[3*rows+i]=cvRound(sum_dst_y/cols);
    };

    //特征相似度判断
    long fenziX=0,SrcPowX=0,SrcPowY=0;
    long fenziY=0,DstPowX=0,DstPowY=0;
    for(size_t i=0;i<rows;i++)
    {
        fenziX+=project[i]*project[2*rows+i];
        fenziY+=project[rows+i]*project[3*rows+i];
        SrcPowX+=project[i]*project[i];
        SrcPowY+=project[rows+i]*project[rows+i];
        DstPowX+=project[2*rows+i]*project[2*rows+i];
        DstPowY+=project[3*rows+i]*project[3*rows+i];
    }
    project.deallocate();        //回收内存空间
    const float eps=0.0001;      //epsilon to avoid division by zero
    float sim=0.5*fenziX/(sqrtf(SrcPowX)*sqrtf(DstPowX)+eps)+0.5*fenziY/(sqrtf(SrcPowY)*sqrtf(DstPowY)+eps);
    sim*=100.00f;
    
    //打印分数值
    Log::Info(m_classId+" secondIdentify match thresh and score: "+to_string(m_secondThres)+","+to_string(sim));
    if(sim>=m_secondThres) return true;
    else return false;    
}

static bool firstIndentify(LaTou* latou,Mat test_img,Mat templ_image,Detector detector,
                            vector<shapeInfo_producer::Info> infos,
                           float& rotateAngle,Mat& outputMat,Point2f& outputCenter)
{
    int m_firstThres=latou->m_firstThres;
    string m_classId=latou->m_classId;
    templ_image=templ_image(latou->m_train.tange()); //模板图像进行裁剪
    cv::Size templ_size=templ_image.size();   //模板图像的尺度
    int padding = 32;                         //padding大于16比较合适
    cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,test_img.cols + 2*padding,test_img.type(),cv::Scalar::all(0));
    test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));  
    int stride = 16;                         //computeResponseMaps函数中使用
    int n = padded_img.rows/stride;
    int m = padded_img.cols/stride;
    Rect roi(0, 0, stride*m , stride*n);    //舍弃边缘东西
    Mat img = padded_img(roi).clone();
    assert(img.isContinuous());             //判断图像连续
    auto matches = detector.match(img,m_firstThres,{m_classId});  
    outputMat=img.clone();                  //进行二次判断的

    //日志
    float score=matches.size()>0 ? matches[0].similarity:0;
    Log::Info(m_classId+" firstIndentify match thresh and score:"+to_string(m_firstThres)+","+to_string(score));
    //每张图片上只有一个物体，找到匹配值最大值的
    if(matches.size()&&(matches[0].similarity)>=m_firstThres)
    {
        auto match = matches[0];                                 //排在第一个就是分数值最大的
        auto templ = detector.getTemplates(m_classId,match.template_id);
        shapeInfo_producer::Info info=infos[match.template_id];  //匹配信息
        float angle=info.angle;                                  //角度
        // if(m_classId=="lapian")
        // {
        //     for(int i=0; i<templ[0].features.size(); i++)
        //     {
        //         auto feat = templ[0].features[i];
        //         cv::circle(img, {feat.x+match.x, feat.y+match.y},1,CV_RGB(0,255,0), -1);
        //     }
        //     std::cout<<"angle:"<<angle<<std::endl;
        //     cv::imshow("img",img);
        //     cv::waitKey(0);
        // }
        float x =  match.x - templ[0].tl_x + templ_size.width/2.0+50;
        float y =  match.y - templ[0].tl_y + templ_size.height/2.0+50;
        // float rx_scaled = templ_size.width*info.scale;   //中心点坐标
        // float ry_scaled = templ_size.height*info.scale;
        // cv::RotatedRect rotatedRectangle({x, y}, {rx_scaled, ry_scaled}, -angle);
        // cv::Point2f vertices[4];
        // rotatedRectangle.points(vertices);
        // for(int i=0; i<4; i++)
        // {
        //     int next = (i+1==4) ? 0 : (i+1);
        //     cv::line(img, vertices[i], vertices[next], CV_RGB(0,0,255), 1);
        // }
        // cv::imshow("img",img);
        // cv::waitKey(0);

        //用于二次判断
        rotateAngle=angle;
        outputCenter=Point2f(x,y);
        return true;
    }
    else
        return false;
};

static int indentify(LaTou* latou,Mat test_img,Mat templ_image,Detector detector,vector<shapeInfo_producer::Info> infos)
{
    //一次判断
    float   rotateAngle;
    Mat     outputMat;
    Point2f outputCenter;
    string m_classId=latou->m_classId; 
    int m_isSec=latou->m_isSec;
    bool first=firstIndentify(latou,test_img,templ_image,detector,infos,rotateAngle,outputMat,outputCenter);
    if(!first)
    {
        Log::Info(m_classId+" firstIndentify is:ng");
        return 3;
    }
    Log::Info(m_classId+" firstIndentify is:ok");
    if(!m_isSec) return 2;

    //进行二次判断
    first=latou->secondIdentify(templ_image,outputMat,outputCenter,rotateAngle);
    if(!first)
    {
        Log::Info(m_classId+" secondIndentify is:ng");
        return 3;
    }
    Log::Info(m_classId+" secondIndentify is:ok");
    return 2;
};


//blank:1,ok:2,false:3
int LaTou::predict(Mat test_img,Mat templ_image,Mat background)
{
    if(m_classId=="dizuo")
    {
        static Detector dizuo_detector;
        static vector<shapeInfo_producer::Info> dizuo_infos;
        if(!m_isLoad)
        {
            dizuo_detector=Detector(m_numFeatures,{4,8},m_weakThreshold,m_strongThreshold);
            dizuo_detector.readClasses({m_classId}, m_imagefolder+"%s.yaml");
            dizuo_infos = shapeInfo_producer::load_infos(m_imagefolder+m_classId+"info.yaml");
            if(dizuo_infos.size()) Log::Info("load "+m_classId+" info file successfully");
            getSegRegion(background);        //针对背景空图计算进行更换
            m_isLoad=true;
        }
        if(!segDizuo(test_img,background))   //blank测试
        {
            Log::Info(m_classId+" blank res is:ok");
            return 1;
        }
        Log::Info(m_classId+" blank res is:ng");
        return indentify(this,test_img,templ_image,dizuo_detector,dizuo_infos);
    }
    if(m_classId=="lapian")
    {
        static Detector lapian_detector;
        static vector<shapeInfo_producer::Info> lapian_infos;
        if(!m_isLoad)
        {
            lapian_detector=Detector(m_numFeatures,{4,8},m_weakThreshold,m_strongThreshold);
            lapian_detector.readClasses({m_classId}, m_imagefolder+"%s.yaml");
            lapian_infos = shapeInfo_producer::load_infos(m_imagefolder+m_classId+"info.yaml");
            if(lapian_infos.size()) Log::Info("load "+m_classId+" info file successfully");
            m_isLoad=true;
        }
        return indentify(this,test_img,templ_image,lapian_detector,lapian_infos);
    }  
}
//创建log日志
void Interface::createLog(const string path)
{
    const int TIMESTAMP_BUFFER_SIZE = 21;
	char buffer[TIMESTAMP_BUFFER_SIZE];
	time_t timestamp;
	time( &timestamp );
	strftime(buffer, sizeof(buffer), "%Y-%m-%d", localtime(&timestamp));
    string name=path+string(buffer)+".txt";
    Log::Initialise(name); 
}

//
int Interface::train(string imagefolder,int is_sec,ROISAB dizuo_train,ROISAB dizuo_logo,ROISAB lapian_train,ROISAB lapian_logo)
{
    //开始时，模型都没有
    m_dizuo.m_isLoad=false;
    m_lapian.m_isLoad=false;

    Timer timer;
    //dizuo
    Mat dizuo_template=imread(imagefolder+"dizuo_template.jpg");
    Mat no_dizuo=imread(imagefolder+"no_dizuo.jpg");
    if(dizuo_template.empty()||no_dizuo.empty()) 
        Log::Error("dizuo_template or no_dizuo empty in function train!");
    string id="dizuo";
    m_dizuo.set(imagefolder,id);
    m_dizuo.loadDetector();
    m_dizuo.m_isSec=is_sec;
    m_dizuo.m_train=dizuo_train;       //训练区域
    m_dizuo.m_logo=dizuo_logo;
    m_dizuo.writeDetector();
    //进行画图
    {
        Mat dizuo_template_roi=dizuo_template.clone();
        rectangle(dizuo_template_roi,dizuo_train.tange(),CV_RGB(255,0,0),1);
        rectangle(dizuo_template_roi,dizuo_logo.tange(),CV_RGB(255,0,0),1);
        imwrite(imagefolder+"dizuo_template_roi.jpg",dizuo_template_roi);
    }
    bool resdz=m_dizuo.train(dizuo_template,no_dizuo);

    //lapian
    Mat lapian_template=imread(imagefolder+"lapian_template.jpg");
    Mat no_lapian=imread(imagefolder+"no_lapian.jpg");
    if(lapian_template.empty()||no_lapian.empty()) 
        Log::Error("lapian_template or no_lapian empty in function train!");
    id="lapian";
    m_lapian.set(imagefolder,id);  //构造对象
    m_lapian.loadDetector();
    m_lapian.m_isSec=is_sec;
    m_lapian.m_train=lapian_train;
    m_lapian.m_logo=lapian_logo;
    m_lapian.writeDetector();
    //进行画图
    {
        Mat lapian_template_roi=lapian_template.clone();          //画框结果显示
        rectangle(lapian_template_roi,lapian_train.tange(),CV_RGB(255,0,0),1);
        rectangle(lapian_template_roi,lapian_logo.tange(),CV_RGB(255,0,0),1);
        imwrite(imagefolder+"lapian_template_roi.jpg",lapian_template_roi);
    }
    bool reslp=m_lapian.train(lapian_template,no_lapian);
    Log::Info("train cost time: "+std::to_string(timer.out()));
    
    int res=4;
    if(resdz==true&&reslp==true)
        res=1;
    else if(resdz==false&&reslp==true)
        res=2;
    else if(resdz==true&&reslp==false)
        res=3;
    else if(resdz==false&&reslp==false)
        res=4;
    return res;
};

int Interface::predict(string imagefolder)
{
     Timer timer;
    //dizuo
    Mat dizuo_image=imread(imagefolder+"dizuo.jpg");
    Mat dizuo_template=imread(imagefolder+"dizuo_template.jpg");
    Mat no_dizuo=imread(imagefolder+"no_dizuo.jpg");
    string id="dizuo";
    m_dizuo.set(imagefolder,id);
    m_dizuo.loadDetector();
    int dizuo_reg=m_dizuo.predict(dizuo_image,dizuo_template,no_dizuo);
    if(dizuo_reg==1) return dizuo_reg;  // 1为空白图

    //lapian
    Mat lapian_image=imread(imagefolder+"lapian.jpg");
    Mat lapian_template=imread(imagefolder+"lapian_template.jpg");
    Mat no_lapian=imread(imagefolder+"no_lapian.jpg");
    id="lapian";
    m_lapian.set(imagefolder,id);  //构造对象
    m_lapian.loadDetector();
    int lapian_reg=m_lapian.predict(lapian_image,lapian_template,no_lapian);
    double predicttime=timer.out();
    Log::Info("predict cost time: "+to_string(predicttime));
    return dizuo_reg|lapian_reg;  //2为OK,3为ng
}

//批量测试图片
void Interface::testpicture(string imagefolder,string dizuopath,string lapianpath)
{
    Timer timer;
    //dizuo
    Mat dizuo_image=imread(dizuopath);
    Log::Info("dizuopath:"+dizuopath);
    Mat dizuo_template=imread(imagefolder+"dizuo_template.jpg");
    Mat no_dizuo=imread(imagefolder+"no_dizuo.jpg");
    string id="dizuo";
    m_dizuo.set(imagefolder,id);
    m_dizuo.loadDetector();
    int dizuo_reg=m_dizuo.predict(dizuo_image,dizuo_template,no_dizuo);
    if(dizuo_reg==1) return ;

    //lapian
    Mat lapian_image=imread(lapianpath);
    Log::Info("lapianpath:"+lapianpath);
    Mat lapian_template=imread(imagefolder+"lapian_template.jpg");
    Mat no_lapian=imread(imagefolder+"no_lapian.jpg");
    id="lapian";
    m_lapian.set(imagefolder,id);  //构造对象
    m_lapian.loadDetector();
    int lapian_reg=m_lapian.predict(lapian_image,lapian_template,no_lapian);
    double predicttime=timer.out();
    Log::Info("predict cost time: "+to_string(predicttime));
};


void Interface::draw(string imagefolder,int is_sec,ROISAB dizuo_train,ROISAB dizuo_logo,ROISAB lapian_train,ROISAB lapian_logo)
{
    //  Timer timer;
    // //dizuo
    // Mat dizuo_template=imread(imagefolder+"dizuo_template.jpg");
    // Mat no_dizuo=imread(imagefolder+"no_dizuo.jpg");
    // if(dizuo_template.empty()||no_dizuo.empty()) 
    //     Log::Error("dizuo_template or no_dizuo empty in function train!");
    // string id="dizuo";
    // LaTou dizuo;     //构造对象
    // dizuo.set(imagefolder,id);
    // dizuo.loadDetector();
    // dizuo.m_isSec=is_sec;
    // dizuo.m_train=dizuo_train;       //训练区域
    // dizuo.m_logo=dizuo_logo;
    // dizuo.writeDetector();
    // //进行画图
    // {
    //     Mat dizuo_template_roi=dizuo_template.clone();
    //     rectangle(dizuo_template_roi,dizuo_train.tange(),CV_RGB(255,0,0),1);
    //     rectangle(dizuo_template_roi,dizuo_logo.tange(),CV_RGB(255,0,0),1);
    //     imwrite(imagefolder+"dizuo_template_roi.jpg",dizuo_template_roi);
    // }
    // //dizuo.train(dizuo_template,no_dizuo);

    // //lapian
    // Mat lapian_template=imread(imagefolder+"lapian_template.jpg");
    // Mat no_lapian=imread(imagefolder+"no_lapian.jpg");
    // if(lapian_template.empty()||no_lapian.empty()) 
    //     Log::Error("lapian_template or no_lapian empty in function train!");
    // id="lapian";
    // LaTou lapian;
    // lapian.set(imagefolder,id);  //构造对象
    // lapian.m_isSec=is_sec;
    // lapian.m_train=lapian_train;
    // lapian.m_logo=lapian_logo;
    // lapian.writeDetector();
    // //进行画图
    // {
    //     Mat lapian_template_roi=lapian_template.clone();          //画框结果显示
    //     rectangle(lapian_template_roi,lapian_train.tange(),CV_RGB(255,0,0),1);
    //     rectangle(lapian_template_roi,lapian_logo.tange(),CV_RGB(255,0,0),1);
    //     imwrite(imagefolder+"lapian_template_roi.jpg",lapian_template_roi);
    // }
    // //lapian.train(lapian_template,no_lapian);
    // Log::Info("train cost time: "+std::to_string(timer.out()));
}

///media/gzy/f093187a-674a-48c6-8baf-b04e5e3a9a0a/latou/latouAlgorithm/temporal_latou 1 1 50 26 326 282 208 78 258 234 470 134 792 384 464 206 522 304
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
    Interface::createLog(imagefolder);
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

    // int tr=interface.train(imagefolder,is_sec,dizuo_train,dizuo_logo,lapian_train,lapian_logo);
    // std::cout<<tr<<std::endl;

    std::cout<<interface.predict(imagefolder)<<std::endl;
    int a=0;
    while(a++<10)
    {
        std::cout<<interface.predict(imagefolder)<<std::endl;
    }
    return 0;
}

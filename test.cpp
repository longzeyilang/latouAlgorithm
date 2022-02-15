#include "shapeMatch.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

static std::string prefix = "/home/gzy/latou/shape_based_matching/lapian/";

// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

//得到所有文件
void getFiles(string path,vector<string>& files)
{
    ifstream infile;
    infile.open(path.data());  //将文件流对象与文件连接起来
    assert(infile.is_open());
    string s;
    while(getline(infile,s))
    {
        files.emplace_back(s);
    }
    infile.close();            //关闭文件输入流
}


void circle_gen(){
    Mat bg = Mat(800, 800, CV_8UC3, {0, 0, 0});
    cv::circle(bg, {400, 400}, 200, {255,255,255}, -1);
    cv::imshow("test", bg);
    waitKey(0);
}

void scale_test(string mode = "train"){
    int num_feature = 150;

    // feature numbers(how many ori in one templates?)
    // two pyramids, lower pyramid(more pixels) in stride 4, lower in stride 8
    shapeMatch::Detector detector(num_feature, {4, 8});

//    mode = "test";
    if(mode == "train"){
        Mat img = cv::imread(prefix+"case0/templ/circle.png");
        assert(!img.empty() && "check your img path");
        shapeMatch::shapeInfo_producer shapes;
        shapes.set(img);
        shapes.scale_range = {0.1f, 1};
        shapes.scale_step = 0.01f;
        shapes.produce_infos();

        std::vector<shapeMatch::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "circle";
        for(auto& info: shapes.infos){

            // template img, id, mask,
            //feature numbers(missing it means using the detector initial num)
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info),
                                                int(num_feature*info.scale));
            std::cout << "templ_id: " << templ_id << std::endl;

            // may fail when asking for too many feature_nums for small training img
            if(templ_id != -1){  // only record info when we successfully add template
                infos_have_templ.push_back(info);
            }
        }

        // save templates
        detector.writeClasses(prefix+"case0/%s_templ.yaml");

        // save infos,
        // in this simple case infos are not used
        shapes.save_infos(infos_have_templ, prefix + "case0/circle_info.yaml");
        std::cout << "train end" << std::endl << std::endl;

    }else if(mode=="test"){
        std::vector<std::string> ids;

        // read templates
        ids.push_back("circle");
        detector.readClasses(ids, prefix+"case0/%s_templ.yaml");

        Mat test_img = imread(prefix+"case0/1.jpg");
        assert(!test_img.empty() && "check your img path");

        // make the img having 32*n width & height
        // at least 16*n here for two pyrimads with strides 4 8
        int stride = 32;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        Timer timer;
        // match, img, min socre, ids
        auto matches = detector.match(img, 75, ids);
        // one output match:
        // x: top left x
        // y: top left y
        // template_id: used to find templates
        // similarity: scores, 100 is best
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++){
            auto match = matches[i];
            auto templ = detector.getTemplates("circle",
                                               match.template_id);
            // template:
            // nums: num_pyramids * num_modality (modality, depth or RGB, always 1 here)
            // template[0]: lowest pyrimad(more pixels)
            // template[0].width: actual width of the matched template
            // template[0].tl_x / tl_y: topleft corner when cropping templ during training
            // In this case, we can regard width/2 = radius
            int x =  templ[0].width/2 + match.x;
            int y = templ[0].height/2 + match.y;
            int r = templ[0].width/2;
            Scalar color(255, rand()%255, rand()%255);

            cv::putText(img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, color);
            cv::circle(img, {x, y}, r, color, 2);
        }

        imshow("img", img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void angle_test(string mode = "train")
{
    shapeMatch::Detector detector(128, {4, 8});
    if(mode == "train")
    {
        Mat img = imread(prefix+"case1/train.png");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shapeMatch::shapeInfo_producer shapes;
        shapes.set(padded_img, padded_mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shapeMatch::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos)
        {
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }
    else if(mode=="test")
    {
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml");

        // angle & scale are saved here, fetched by match id
        auto infos = shapeMatch::shapeInfo_producer::load_infos(prefix + "case1/test_info.yaml");

        Mat test_img = imread(prefix+"case1/test.png");
        assert(!test_img.empty() && "check your img path");

        int padding = 500;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());
//        cvtColor(img, img, CV_BGR2GRAY);
        std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;
        Timer timer;
        auto matches = detector.match(img, 90, ids);
        timer.out();
        if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 1;
        if(top5>matches.size()) top5=matches.size();
        for(size_t i=0; i<top5; i++)
        {
            auto match = matches[i];
            auto templ = detector.getTemplates("test",match.template_id);

            // 270 is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training

            float r_scaled = 270/2.0f*infos[match.template_id].scale;

            // scaling won't affect this, because it has been determined by warpAffine
            // cv::warpAffine(src, dst, rot_mat, src.size()); last param
            float train_img_half_width = 270/2.0f + 100;

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_width;

            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;
            for(int i=0; i<templ[0].features.size(); i++)
            {
                auto feat = templ[0].features[i];
                cv::circle(img, {feat.x+match.x, feat.y+match.y}, 3, randColor, -1);
            }
            cv::putText(img, to_string(int(round(match.similarity))),Point(match.x+r_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::RotatedRect rotatedRectangle({x, y}, {2*r_scaled, 2*r_scaled}, -infos[match.template_id].angle);
            cv::Point2f vertices[4];
            rotatedRectangle.points(vertices);
            for(int i=0; i<4; i++)
            {
                int next = (i+1==4) ? 0 : (i+1);
                cv::line(img, vertices[i], vertices[next], randColor, 2);
            }
            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }
        imshow("img", img);
        waitKey(0);
        std::cout << "test end" << std::endl << std::endl;
    }
}


void getFiles( string path, vector<string>& files )  
{  
    ifstream infile; 
    infile.open(path.data());   //将文件流对象与文件连接起来 
    assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
    string s;
    while(getline(infile,s))
    {
       // cout<<s<<endl;
       files.push_back(s);
    }
    infile.close();             //关闭文件输入流 
}  

void angle_test_lapian(string mode = "train")
{
    shapeMatch::Detector detector(128, {4, 8},20,50);
    if(mode == "train")
    {
        Mat img = imread(prefix+"Image_20190127134057183.jpg");
        assert(!img.empty() && "check your img path");
        Rect roi(366, 230, 680, 666);
        img = img(roi).clone();
        cv::imwrite(prefix+"templ.bmp",img);
//        if(img.channels()==3) cv::cvtColor(img,img,CV_BGR2GRAY);
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shapeMatch::shapeInfo_producer shapes;
        shapes.set(padded_img, padded_mask);
        shapes.angle_range = {-5,5};
        shapes.angle_step = 1;
        shapes.scale_range={0.9,1.1};
        shapes.scale_step=0.05;
        shapes.produce_infos();
        std::vector<shapeMatch::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos)
        {
            imshow("train", shapes.src_of(info));
            waitKey(1);
            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1) infos_have_templ.push_back(info);
        }
        detector.writeClasses(prefix+"%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }
    else if(mode=="test")
    {
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"%s_templ.yaml");
        // angle & scale are saved here, fetched by match id
        auto infos = shapeMatch::shapeInfo_producer::load_infos(prefix + "test_info.yaml");
        std::vector<std::string> filenames;
        //读取txt文件
        getFiles("/home/gzy/latou/shape_based_matching/path.txt",filenames);
        for(int k=0;k<filenames.size();k++)
        {
            std::string fileFullPath=prefix+filenames[k];
            Mat test_img = imread(fileFullPath);
            assert(!test_img.empty() && "check your img path");
            int padding = 0;
            cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
            test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

            int stride = 16;        //computeResponseMaps函数中使用
            int n = padded_img.rows/stride;
            int m = padded_img.cols/stride;
            Rect roi(0, 0, stride*m , stride*n);
            Mat img = padded_img(roi).clone();
            assert(img.isContinuous());
//        cvtColor(img, img, CV_BGR2GRAY);
            std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;
            Timer timer;
            auto matches = detector.match(img, 85, ids);
            timer.out();
            if(img.channels() == 1) cvtColor(img, img, CV_GRAY2BGR);
            std::cout << "matches.size(): " << matches.size() << std::endl;
            size_t top5 = 1;
            if(top5>matches.size()) top5=matches.size();
            for(size_t i=0; i<top5; i++)
            {
                auto match = matches[i];
                auto templ = detector.getTemplates("test",match.template_id);
                // 270 is width of template image
                // 100 is padding when training
                // tl_x/y: template croping topleft corner when training
                float rx_scaled = 680/2.0f*infos[match.template_id].scale;
                float ry_scaled = 666/2.0f*infos[match.template_id].scale;
    
                // scaling won't affect this, because it has been determined by warpAffine
                // cv::warpAffine(src, dst, rot_mat, src.size()); last param
                float train_img_half_width = 680/2.0f + 100;
                float train_img_half_height = 666/2.0f + 100;
    
                // center x,y of train_img in test img
                float x =  match.x - templ[0].tl_x + train_img_half_width;
                float y =  match.y - templ[0].tl_y + train_img_half_height;
    
 //               cv::Vec3b randColor=cv::Scalar(0,0,255);
//                randColor[0] = rand()%155 + 100;
//                randColor[1] = rand()%155 + 100;
//                randColor[2] = rand()%155 + 100;
                for(int i=0; i<templ[0].features.size(); i++)
                {
                    auto feat = templ[0].features[i];
                    cv::circle(img, {feat.x+match.x, feat.y+match.y},3, cv::Scalar(0,255,0), -1);
                }
    
                cv::putText(img, to_string(int(round(match.similarity))),Point(match.x+rx_scaled-10, match.y-3), FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
                cv::RotatedRect rotatedRectangle({x, y}, {2*rx_scaled, 2*ry_scaled}, -infos[match.template_id].angle);
                cv::Point2f vertices[4];
                rotatedRectangle.points(vertices);
                for(int i=0; i<4; i++)
                {
                    int next = (i+1==4) ? 0 : (i+1);
                    cv::line(img, vertices[i], vertices[next], cv::Scalar(0,0,255), 2);
                }
                std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
                std::cout << "match.similarity: " << match.similarity << std::endl;
            }
            imshow("img", img);
            waitKey(0);
            std::cout << "test end" << std::endl << std::endl;
        }
    }
}


void noise_test(string mode = "train")
{
    shapeMatch::Detector detector(30, {4, 8});
//    mode = "test";
    if(mode == "train")
    {
        Mat img = imread(prefix+"case2/train.png");
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shapeMatch::shapeInfo_producer shapes;
        shapes.set(img, mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shapeMatch::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos){
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case2/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case2/test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(ids, prefix+"case2/%s_templ.yaml");

        Mat test_img = imread(prefix+"case2/test.png");
        assert(!test_img.empty() && "check your img path");

        // cvtColor(test_img, test_img, CV_BGR2GRAY);

        int stride = 16;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        test_img = test_img(roi).clone();

        Timer timer;
        auto matches = detector.match(test_img, 80, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;
            auto templ = detector.getTemplates("test",match.template_id);
            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            int x =  templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imshow("img", test_img);
        waitKey(0);

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

void view_angle(){
    float weak_thresh = 30.0f;

    // default params for detector
    shapeMatch::Detector detector(63, {4, 8}, weak_thresh, 60.0f);
    // last two: magnitude thresh to extract angle in test image;
    //magnitude thresh to extract template points in train image;

    Mat img = cv::imread(prefix+"case0/templ/circle.png");
    assert(!img.empty() && "check your img path");
    imshow("img", img);
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_BGR2GRAY);

    GaussianBlur(gray, gray, {5, 5}, 0);

    Mat grad1,grad2,angle;
    Sobel(gray, grad1, CV_32FC1, 1, 0);
    Sobel(gray, grad2, CV_32FC1, 0, 1);
    phase(grad1, grad2, angle, true);
    for(int r=0; r<angle.rows; r++){
        for(int c=0; c<angle.cols; c++){
            if(angle.at<float>(r, c) > 180)
            angle.at<float>(r, c) -= 180;
        }
    }
    angle.convertTo(angle,CV_8UC1);
    Mat grad_mask = (grad1.mul(grad1) + grad2.mul(grad2)) > weak_thresh*weak_thresh;
    Mat angle_masked;
    angle.copyTo(angle_masked, grad_mask);
    imshow("mask", grad_mask);
    imshow("angle", angle_masked);
    // angle masked is what we use for shape based matching
    cv::waitKey(0);
}

int main()
{
    MIPP_test();
    //angle_test_lapian();
    return 0;
}

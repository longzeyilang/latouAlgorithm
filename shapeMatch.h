#ifndef SHAPEMATCH_H
#define SHAPEMATCH_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include <map>
#include "mipp.h"  // for SIMD in different platforms
#include <chrono>
#include "Log.h"
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { return std::chrono::duration_cast<second_>(clock_::now() - beg_).count(); }
    double out()
    {
        double t = elapsed();
        reset();
        return t;
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};

struct Feature
{
    int x;         //x,y代表特征点坐标
    int y;
    int label;     //label代表0～7
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
    Feature(): x(0), y(0), label(0) {}
    Feature(int _x, int _y, int _label): x(_x), y(_y), label(_label) {};
};

struct Template
{
    int width;
    int height;
    int tl_x;                        //矩形左上角点坐标最小值
    int tl_y;                        //矩形左上角点坐标最小值
    int pyramid_level;               //金字塔层级
    std::vector<Feature> features;   //同一个金字塔层级，所有（angle，scale）的特征点数
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
};

class ColorGradientPyramid
{
public:
    ColorGradientPyramid(const cv::Mat &src, const cv::Mat &mask,float weak_threshold,size_t num_features,float strong_threshold);
    void quantize(cv::Mat &dst) const;
    bool extractTemplate(Template &templ) const;
    void pyrDown();
    void update();
    /// Candidate feature with a score
    struct Candidate
    {
        /// Sort candidates with high score to the front
        bool operator<(const Candidate &rhs) const
        {
            return score > rhs.score;
        }
        Feature f;
        float score;   //梯度值
        //x,y代表特征点的坐标值，label代表0～7之间标签值，score_代表特征点的梯度值
        Candidate(int x, int y, int label, float score_):f(x,y,label),score(score_){};
    };
    cv::Mat src;
    cv::Mat mask;
    int pyramid_level;
    cv::Mat angle;
    cv::Mat magnitude;
    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    static bool selectScatteredFeatures(const std::vector<Candidate> &candidates,std::vector<Feature> &features,size_t num_features, float distance);
};


class ColorGradient
{
public:
    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);
    std::string name() const;
    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
    cv::Ptr<ColorGradientPyramid> process(const cv::Mat src, const cv::Mat &mask = cv::Mat()) const
    {
        return cv::makePtr<ColorGradientPyramid>(src, mask, weak_threshold, num_features, strong_threshold);
    }
};

//匹配点
struct Match
{
    Match(){}
    Match(int _x, int _y, float _similarity, const std::string &_class_id, int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
    {

    };
    bool operator<(const Match &rhs) const
    {
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }
    bool operator==(const Match &rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }
    int x;
    int y;
    float similarity;
    std::string class_id;
    int template_id;
};

class Detector
{
public:
    /**
         * \brief Empty constructor, initialize with read().
    */
    Detector();
    Detector(std::vector<int> T);
    Detector(int num_features, std::vector<int> T, float weak_thresh = 30.0f, float strong_thresh = 60.0f);
    std::vector<Match> match(cv::Mat sources, int threshold,const std::vector<std::string> &class_ids = std::vector<std::string>(),const cv::Mat masks = cv::Mat()) const;
    int addTemplate(const cv::Mat sources, const std::string &class_id,const cv::Mat &object_mask,int num_features = 0);
    const cv::Ptr<ColorGradient> &getModalities() const { return modality; }
    int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }
    int pyramidLevels() const { return pyramid_levels; }
    const std::vector<Template> &getTemplates(const std::string &class_id, int template_id) const;
    int numTemplates() const;
    int numTemplates(const std::string &class_id) const;
    int numClasses() const { return static_cast<int>(class_templates.size()); }
    std::vector<std::string> classIds() const;
    void read(const cv::FileNode &fn);
    void write(cv::FileStorage &fs) const;
    std::string readClass(const cv::FileNode &fn, const std::string &class_id_override = "");
    void writeClass(const std::string &class_id, cv::FileStorage &fs) const;
    void readClasses(const std::vector<std::string> &class_ids,const std::string &format = "templates_%s.yml.gz");
    void writeClasses(const std::string &format = "templates_%s.yml.gz") const;
    void deleteClass(const std::string class_id);
protected:
    cv::Ptr<ColorGradient> modality;
    int pyramid_levels;
    std::vector<int> T_at_level;
    typedef std::vector<Template> TemplatePyramid; //金字塔模板数,每个金字塔层数代表同一层的所有模板数
    typedef std::map<std::string, std::vector<TemplatePyramid>> TemplatesMap;
    TemplatesMap class_templates;
    typedef std::vector<cv::Mat> LinearMemories;
    // Indexed as [pyramid level][ColorGradient][quantized label]
    typedef std::vector<std::vector<LinearMemories>> LinearMemoryPyramid;
    void matchClass(const LinearMemoryPyramid &lm_pyramid,const std::vector<cv::Size> &sizes,
                    int threshold, std::vector<Match> &matches,const std::string &class_id,
                    const std::vector<TemplatePyramid> &template_pyramids) const;
};

class shapeInfo_producer
{
public:
    cv::Mat src;
    cv::Mat mask;
    std::vector<float> angle_range;
    std::vector<float> scale_range;
    float angle_step = 15;
    float scale_step = 0.5;
    float eps = 0.00001f;
    class Info                   //模板图像角度和缩放类
    {
    public:
        float angle;
        float scale;
        Info(float angle_, float scale_)
        {
            angle = angle_;
            scale = scale_;
        }
    };
    std::vector<Info> infos;
    void set(cv::Mat src, cv::Mat mask = cv::Mat())     //设置源图像和掩码图像
    {
        this->src = src;
        if(mask.empty())
        {
            // make sure we have masks
            this->mask = cv::Mat(src.size(), CV_8UC1, {255});
        }
        else
        {
            this->mask = mask;
        }
    }

    static cv::Mat transform(cv::Mat src, float angle, float scale)
    {
        cv::Mat dst;
        cv::Point center(src.cols/2, src.rows/2);
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::warpAffine(src, dst, rot_mat, src.size());
        return dst;
    }
    static void save_infos(std::vector<shapeInfo_producer::Info>& infos, std::string path = "infos.yaml")
    {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        fs << "infos"
           << "[";
        for (int i = 0; i < infos.size(); i++)
        {
            fs << "{";
            fs << "angle" << infos[i].angle;
            fs << "scale" << infos[i].scale;
            fs << "}";
        }
        fs << "]";
    }
    static std::vector<Info> load_infos(std::string path = "info.yaml")
    {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        std::vector<Info> infos;
        cv::FileNode infos_fn = fs["infos"];
        cv::FileNodeIterator it = infos_fn.begin(), it_end = infos_fn.end();
        for (int i = 0; it != it_end; ++it, i++)
        {
            infos.emplace_back(float((*it)["angle"]), float((*it)["scale"]));
        }
        return infos;
    }

    void produce_infos()
    {
        assert(angle_range.size() <= 2);
        assert(scale_range.size() <= 2);
        assert(angle_step > eps*10);
        assert(scale_step > eps*10);

        // make sure range not empty
        if(angle_range.size() == 0)
        {
            angle_range.push_back(0);
        }
        if(scale_range.size() == 0)
        {
            scale_range.push_back(1);
        }
        if(angle_range.size() == 1 && scale_range.size() == 1)
        {
            float angle = angle_range[0];
            float scale = scale_range[0];
            infos.emplace_back(angle, scale);
        }
        else if(angle_range.size() == 1 && scale_range.size() == 2)
        {
            assert(scale_range[1] > scale_range[0]);
            float angle = angle_range[0];
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step)
            {
                infos.emplace_back(angle, scale);
            }
        }
        else if(angle_range.size() == 2 && scale_range.size() == 1)
        {
            assert(angle_range[1] > angle_range[0]);
            float scale = scale_range[0];
            for(float angle = angle_range[0]; angle <= angle_range[1]; angle += angle_step)
            {
                infos.emplace_back(angle, scale);
            }
        }
        else if(angle_range.size() == 2 && scale_range.size() == 2)
        {
            assert(scale_range[1] > scale_range[0]);
            assert(angle_range[1] > angle_range[0]);
            for(float scale = scale_range[0]; scale <= scale_range[1]+eps; scale += scale_step)
            {
                for(float angle = angle_range[0]; angle <= angle_range[1]+eps; angle += angle_step)
                {
                    infos.emplace_back(angle, scale);
                }
            }
        }
    }

    cv::Mat src_of(const Info& info)
    {
        return transform(src, info.angle, info.scale);
    }

    cv::Mat mask_of(const Info& info)
    {
        return (transform(mask, info.angle, info.scale) > 0);
    }
    void clean()
    {
        if(infos.size()) infos.clear();
        if(angle_range.size()) angle_range.clear();
        if(scale_range.size()) scale_range.clear();
    }
};
#endif

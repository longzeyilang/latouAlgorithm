#include "shapeMatch.h"
#include <iostream>
using namespace std;
using namespace cv; 
/**
 * \brief Get the label [0,8) of the single bit set in quantized.
 */
static inline int getLabel(int quantized)
{
    switch (quantized)
    {
    case 1:return 0;
    case 2:return 1;
    case 4:return 2;
    case 8:return 3;
    case 16:return 4;
    case 32:return 5;
    case 64:return 6;
    case 128:return 7;
    default:
        CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
        return -1; //avoid warning
    }
}

void Feature::read(const FileNode &fn)
{
    FileNodeIterator fni = fn.begin();
    fni >> x >> y >> label;
}

void Feature::write(FileStorage &fs) const
{
    fs << "[:" << x << y << label << "]";
}

void Template::read(const FileNode &fn)
{
    width = fn["width"];
    height = fn["height"];
    tl_x = fn["tl_x"];
    tl_y = fn["tl_y"];
    pyramid_level = fn["pyramid_level"];

    FileNode features_fn = fn["features"];
    features.resize(features_fn.size());
    FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
    for (int i = 0; it != it_end; ++it, ++i)
    {
        features[i].read(*it);
    }
}

void Template::write(FileStorage &fs) const
{
    fs << "width" << width;
    fs << "height" << height;
    fs << "tl_x" << tl_x;
    fs << "tl_y" << tl_y;
    fs << "pyramid_level" << pyramid_level;

    fs << "features"
       << "[";
    for (int i = 0; i < (int)features.size(); ++i)
    {
        features[i].write(fs);
    }
    fs << "]"; // features
}

/**
 * \brief Crop a set of overlapping templates from different modalities.
 *
 * \param[in,out] templates Set of templates representing the same object view.
 *
 * \return The bounding box of all the templates in original image coordinates.
 */
static Rect cropTemplates(std::vector<Template> &templates,Mat source)
{
    int min_x = std::numeric_limits<int>::max();
    int min_y = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int max_y = std::numeric_limits<int>::min();
    // First pass: find min/max feature x,y over all pyramid levels and modalities
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];  
        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            int x = templ.features[j].x << templ.pyramid_level;   // templ.features[j].x * 2^templ.pyramid_level
            int y = templ.features[j].y << templ.pyramid_level;
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
        }
    }
    if (min_x % 2 == 1) --min_x;
    if (min_y % 2 == 1) --min_y;

    // Second pass: set width/height and shift all feature positions
    for (int i = 0; i < (int)templates.size(); ++i)
    {
        Template &templ = templates[i];
        templ.width  = (max_x - min_x) >> templ.pyramid_level;
        templ.height = (max_y - min_y) >> templ.pyramid_level;
        templ.tl_x = min_x >> templ.pyramid_level;
        templ.tl_y = min_y >> templ.pyramid_level;
        for (int j = 0; j < (int)templ.features.size(); ++j)
        {
            //模板的中心点是相对于矩形左上角坐标
            templ.features[j].x -= templ.tl_x;   
            templ.features[j].y -= templ.tl_y;
        }
    }
    Rect roi=Rect(min_x, min_y, max_x - min_x, max_y - min_y);
    return roi;
}

bool ColorGradientPyramid::selectScatteredFeatures(const std::vector<Candidate> &candidates,
                                                   std::vector<Feature> &features,size_t num_features, float distance)
{
    features.clear();
    float distance_sq = distance * distance;  //distance_sq大于4
    int i = 0;
    while (features.size() < num_features)
    {
        Candidate c = candidates[i];
        // Add if sufficient distance away from any previously chosen feature
        bool keep = true;
        for (int j = 0; (j < (int)features.size()) && keep; ++j)
        {
            Feature f = features[j];
            keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
        }
        if (keep) features.push_back(c.f);
        if (++i == (int)candidates.size())
        {
            //从头开始搜索，距离减少
            i = 0;
            distance -= 1.0f;
            distance_sq = distance * distance;
            // if (distance < 3)
            // {
            //     // we don't want two features too close
            //     break;
            // }
        }
    }
    if (features.size() == num_features)
    {
        return true;
    }
    else
    {
        Log::Error("selectScatteredFeatures:this templ has no enough features");
        return false;
    }
}

/****************************************************************************************\
*                                                         Color gradient ColorGradient                                                                        *
\****************************************************************************************/
//quantized_angle是?; angle是sobel_angle0~360之间
void hysteresisGradient(Mat &magnitude, Mat &quantized_angle,Mat &angle, float threshold)
{
    // Quantize 360 degree range of orientations into 16 buckets
    // Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
    // for stability of horizontal and vertical features.
    Mat_<unsigned char> quantized_unfiltered;
    angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);
    //Zero out top and bottom rows
    memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
    memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
    // Zero out first and last columns
    for (int r = 0; r < quantized_unfiltered.rows; ++r)
    {
        quantized_unfiltered(r, 0) = 0;
        quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
    }

    // Mask 16 buckets into 8 quantized orientations
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            quant_r[c] &= 7;    //将8～16转化到0～7之间
        }
    }

    // Filter the raw quantized image. Only accept pixels where the magnitude is above some
    // threshold, and there is local agreement on the quantization.
    quantized_angle = Mat::zeros(angle.size(), CV_8U);
    for (int r = 1; r < angle.rows - 1; ++r)
    {
        float *mag_r = magnitude.ptr<float>(r);
        for (int c = 1; c < angle.cols - 1; ++c)
        {
            if (mag_r[c] > threshold)
            {
                // Compute histogram of quantized bins in 3x3 patch around pixel
                int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

                uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
                histogram[patch3x3_row[0]]++;  
                histogram[patch3x3_row[1]]++;   
                histogram[patch3x3_row[2]]++;   

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;   
                histogram[patch3x3_row[1]]++;   
                histogram[patch3x3_row[2]]++;   

                patch3x3_row += quantized_unfiltered.step1();
                histogram[patch3x3_row[0]]++;   
                histogram[patch3x3_row[1]]++;   
                histogram[patch3x3_row[2]]++;   

                // Find bin with the most votes from the patch
                int max_votes = 0;
                int index = -1;
                for (int i = 0; i < 8; ++i)
                {
                    if (max_votes < histogram[i])
                    {
                        index = i;
                        max_votes = histogram[i];
                    }
                }

                // Only accept the quantization if majority of pixels in the patch agree
                static const int NEIGHBOR_THRESHOLD = 5;
                if (max_votes >= NEIGHBOR_THRESHOLD)
                    quantized_angle.at<uchar>(r, c) = uchar(1 << index);
            }
        }
    }
}

static void quantizedOrientations(const Mat &src, Mat &magnitude,Mat &angle, float threshold)
{
    //threshold=weak_threshold
    Mat smoothed;
    // Compute horizontal and vertical image derivatives on all color channels separately
    static const int KERNEL_SIZE = 7;
    // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
    GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
    if(src.channels() == 1)
    {
        Mat sobel_dx, sobel_dy, magnitude, sobel_ag;
        Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
        magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
    }
    else
    {
        magnitude.create(src.size(), CV_32F);
        Size size = src.size();
        Mat sobel_3dx;              // per-channel horizontal derivative
        Mat sobel_3dy;              // per-channel vertical derivative
        Mat sobel_dx(size, CV_32F); // maximum horizontal derivative
        Mat sobel_dy(size, CV_32F); // maximum vertical derivative
        Mat sobel_ag;               // final gradient orientation (unquantized)

        Sobel(smoothed, sobel_3dx, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(smoothed, sobel_3dy, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);

        short *ptrx = (short *)sobel_3dx.data;
        short *ptry = (short *)sobel_3dy.data;
        float *ptr0x = (float *)sobel_dx.data;
        float *ptr0y = (float *)sobel_dy.data;
        float *ptrmg = (float *)magnitude.data;

        const int length1 = static_cast<const int>(sobel_3dx.step1());
        const int length2 = static_cast<const int>(sobel_3dy.step1());
        const int length3 = static_cast<const int>(sobel_dx.step1());
        const int length4 = static_cast<const int>(sobel_dy.step1());
        const int length5 = static_cast<const int>(magnitude.step1());
        const int length0 = sobel_3dy.cols * 3;

        for (int r = 0; r < sobel_3dy.rows; ++r)
        {
            int ind = 0;
            for (int i = 0; i < length0; i += 3)
            {
                // Use the gradient orientation of the channel whose magnitude is largest
                int mag1 = ptrx[i + 0] * ptrx[i + 0] + ptry[i + 0] * ptry[i + 0];
                int mag2 = ptrx[i + 1] * ptrx[i + 1] + ptry[i + 1] * ptry[i + 1];
                int mag3 = ptrx[i + 2] * ptrx[i + 2] + ptry[i + 2] * ptry[i + 2];
                if (mag1 >= mag2 && mag1 >= mag3)
                {
                    ptr0x[ind] = ptrx[i];
                    ptr0y[ind] = ptry[i];
                    ptrmg[ind] = (float)mag1;
                }
                else if (mag2 >= mag1 && mag2 >= mag3)
                {
                    ptr0x[ind] = ptrx[i + 1];
                    ptr0y[ind] = ptry[i + 1];
                    ptrmg[ind] = (float)mag2;
                }
                else
                {
                    ptr0x[ind] = ptrx[i + 2];
                    ptr0y[ind] = ptry[i + 2];
                    ptrmg[ind] = (float)mag3;
                }
                ++ind;
            }
            ptrx += length1;
            ptry += length2;
            ptr0x += length3;
            ptr0y += length4;
            ptrmg += length5;
        }
        // Calculate the final gradient orientations
        phase(sobel_dx, sobel_dy, sobel_ag, true);
        hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);
    }
}

ColorGradientPyramid::ColorGradientPyramid(const Mat &_src, const Mat &_mask,
                                           float _weak_threshold, size_t _num_features,
                                           float _strong_threshold)
    : src(_src),
      mask(_mask),
      pyramid_level(0),
      weak_threshold(_weak_threshold),
      num_features(_num_features),
      strong_threshold(_strong_threshold)
{
    update();
}

//角度量化按弱阈值进行量化
//梯度量化按强阈值进行量化
void ColorGradientPyramid::update()
{
    quantizedOrientations(src, magnitude, angle, weak_threshold);
}

//金字塔进行采样
void ColorGradientPyramid::pyrDown()
{
    num_features /= 2; 
    ++pyramid_level;

    Size size(src.cols / 2, src.rows / 2);
    Mat next_src;
    cv::pyrDown(src, next_src, size);
    src = next_src;
    if (!mask.empty())
    {
        Mat next_mask;
        resize(mask, next_mask, size, 0.0, 0.0, INTER_NEAREST);
        mask = next_mask;
    }
    update();
}

void ColorGradientPyramid::quantize(Mat &dst) const
{
    //angle是量化的quantized_angle 
    dst = Mat::zeros(angle.size(), CV_8U);
    angle.copyTo(dst, mask); 
}

bool ColorGradientPyramid::extractTemplate(Template &templ) const
{
    Mat local_mask=mask.clone();
    // if (!mask.empty())
    // {
    //     erode(mask, local_mask, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);  //腐蚀白色区域
    // }
    std::vector<Candidate> candidates;
    bool no_mask = local_mask.empty();  //false
    float threshold_sq = strong_threshold * strong_threshold;
    int nms_kernel_size = 3;  //由5改成3
    cv::Mat magnitude_valid = cv::Mat(magnitude.size(), CV_8UC1, cv::Scalar(255));
    for (int r = 0+nms_kernel_size/2; r < magnitude.rows-nms_kernel_size/2; ++r)
    {
        const uchar *mask_r = no_mask ? NULL : local_mask.ptr<uchar>(r);
        for (int c = 0+nms_kernel_size/2; c < magnitude.cols-nms_kernel_size/2; ++c)
        {
            if (no_mask || mask_r[c])
            {
                float score = 0;
                if(magnitude_valid.at<uchar>(r, c)>0)
                {
                    score = magnitude.at<float>(r, c);
                    bool is_max = true;
                    //判断3*3领域内是不是最大值
                    for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++)
                    {
                        for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++)
                        {
                            if(r_offset == 0 && c_offset == 0) continue;
                            if(score < magnitude.at<float>(r+r_offset, c+c_offset))
                            {
                                score = 0;
                                is_max = false;
                                break;
                            }
                        }
                    }
                    
                    //如果是最大值，其他位置的梯度都是0
                    if(is_max)
                    {
                        for(int r_offset = -nms_kernel_size/2; r_offset <= nms_kernel_size/2; r_offset++)
                        {
                            for(int c_offset = -nms_kernel_size/2; c_offset <= nms_kernel_size/2; c_offset++)
                            {
                                if(r_offset == 0 && c_offset == 0) continue;
                                magnitude_valid.at<uchar>(r+r_offset, c+c_offset) = 0;
                            }
                        }
                    }
                }
                if (score > threshold_sq && angle.at<uchar>(r, c) > 0)   //angle已经经过量化的
                {
                    candidates.push_back(Candidate(c, r, getLabel(angle.at<uchar>(r, c)), score));
                }
            }
        }
    }
    if (candidates.size() < num_features) return false;
    //score也就是梯度值大的放在前面
    std::stable_sort(candidates.begin(), candidates.end());
    // Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
    float distance = static_cast<float>(candidates.size() / num_features + 1);  //distance>2
    if (!selectScatteredFeatures(candidates, templ.features, num_features, distance))
    {
        return false;
    }
    // Size determined externally, needs to match templates for other modalities
    templ.width = -1;
    templ.height = -1;
    templ.pyramid_level = pyramid_level;
    return true;
}

ColorGradient::ColorGradient(): weak_threshold(30.0f),num_features(63),strong_threshold(60.0f)
{
}

ColorGradient::ColorGradient(float _weak_threshold, size_t _num_features, float _strong_threshold)
    : weak_threshold(_weak_threshold),num_features(_num_features),strong_threshold(_strong_threshold)
{
}

static const char CG_NAME[] = "ColorGradient";
std::string ColorGradient::name() const
{
    return CG_NAME;
}

void ColorGradient::read(const FileNode &fn)
{
    String type = fn["type"];
    CV_Assert(type == CG_NAME);
    weak_threshold = fn["weak_threshold"];
    num_features = int(fn["num_features"]);
    strong_threshold = fn["strong_threshold"];
}

void ColorGradient::write(FileStorage &fs) const
{
    fs << "type" << CG_NAME;
    fs << "weak_threshold" << weak_threshold;
    fs << "num_features" << int(num_features);
    fs << "strong_threshold" << strong_threshold;
}
/****************************************************************************************\
*                                                                 Response maps                                                                                    *
\****************************************************************************************/

static void orUnaligned8u(const uchar *src, const int src_stride,uchar *dst, 
                          const int dst_stride,const int width, const int height)
{
    /*
    {
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,
        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
        8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
        8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
        &src.at<uchar>(0,1)作为src输入时,c为15
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) 
        {
            dst[c]|= src[c];  
            c++;
        }
    };
    */
    for (int r = 0; r < height; ++r)
    {
        int c = 0;
        //16位进行对齐,T为4,c分别为0,15,14,13,移动多少位可以对齐,src偏移c位可以对齐
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) 
        {
            dst[c]|= src[c];  
            c++;
        }
        //width-mipp::N<uint8_t>():16
        for (; c <= width-mipp::N<uint8_t>(); c+=mipp::N<uint8_t>())
        {
            mipp::Reg<uint8_t> src_v((uint8_t*)src + c);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst + c);
            mipp::Reg<uint8_t> res_v = mipp::orb(src_v, dst_v);  //连续16个数进行对应位置元素相加
            res_v.store((uint8_t*)dst + c);                      //dst的值变成res_v
        }
        for(; c<width; c++) dst[c]|= src[c];
        src += src_stride;  //src_stride代表每一行数据
        dst += dst_stride;
    }
}

/**
 * \brief Spread binary labels in a quantized image.
 * \param[in]  src The source 8-bit quantized image. 1,2,4,8,16,32,64,128
 * \param[out] dst Destination 8-bit spread image.
 * \param      T   Sampling step. Spread labels T/2 pixels in each direction.
 */
static void spread(const Mat &src, Mat &dst, int T)
{
    dst = Mat::zeros(src.size(), CV_8U);
    /*
    T={4,8} 
    以T=4为例
    dst[0][0]=src[0][0]|src[0][1]|src[0][2]|src[0][3]|
              src[1][0]|src[1][1]|src[1][2]|src[1][3]|
              src[2][0]|src[2][1]|src[2][2]|src[2][3]|
              src[3][0]|src[3][1]|src[3][2]|src[3][3]|
    */
    for (int r = 0; r < T; ++r)    
        for (int c = 0; c < T; ++c)
        {
            //src.step1():cols*channels  
            orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()),   
                          dst.ptr(),static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
        }  
}

/**
 * \brief Precompute response maps for a spread quantized image.
 *
 * Implements section 2.4 "Precomputing Response Maps."
 *
 * \param[in]  src           The source 8-bit spread quantized image.
 * \param[out] response_maps Vector of 8 response maps, one for each bit label.
 * 
 * 特征点匹配的时候打分有问题:一个特征点就是三个量,x,y坐标跟一个8位的方向(180/8).匹配的时候,如果同样位置上的方向一致,加4分;
 * 方向差22.5度加3;方向差45度加2;方向差67.5度加1;方向差90度加0;就算乱给方向,得分的期望2分,占最高分的一半,给分这么宽容难怪阈值低的
 * 时候出现这么多错误匹配.比如遮挡物体的1/3,现在的实际得分应该是67.但是,由于给分过于宽松,随机物体的分数可以认为是50分的一个正太分布,要
 * 超过67分实在是太容易,所以降低阈值虽然能把部分遮挡的物体检测到,同时会引入大量无匹配.
 * 
 * 一个自然的想法就是把误匹配的分布压下去:方向一致给4分,差22.5度给1分,不然不给分.为什么要这样给,因为这样一个特征点的期望得分是1分,随机物体
 * 的期望得分是25分.
 */
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 
                                                  0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                                  0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                                  0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
                                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 
                                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 
                                                  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};
//src像素点最大值255
static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    //将像素点分为低四位lsb4，高四位msb4
    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);
        for (int c = 0; c < src.cols; ++c)
        {
            //1个字节8bit
            lsb4_r[c] = src_r[c] & 15;           //低四位（右边）
            msb4_r[c] = (src_r[c] & 240) >> 4;   //高四位（左边）
        }
    }

    {
        //拆分成两部分数据
        uchar *lsb4_data = lsb4.ptr<uchar>();
        uchar *msb4_data = msb4.ptr<uchar>();

        bool no_max = true;
        bool no_shuff = true;

        #ifdef has_max_int8_t
        no_max = false;
        #endif

        #ifdef has_shuff_int8_t
        no_shuff = false;
        #endif
        // LUT is designed for 128 bits SIMD, so quite triky for others
        // For each of the 8 quantized orientations...
        for (int ori = 0; ori < 8; ++ori)
        {
            //no_max:false, no_shuff:false, mipp::N<uint8_t>():16
            //8个方向的map
            uchar *map_data = response_maps[ori].ptr<uchar>();
            const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
            if(mipp::N<uint8_t>() == 1 || no_max || no_shuff)           //no SIMD
            { 
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
            else if(mipp::N<uint8_t>() == 16) //128 SIMD, no add base
            { 
                const uchar *lut_low = SIMILARITY_LUT + 32 * ori;
                mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);         //查找表低位，1111对应最大值索引15
                mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16);   //查找表高位，1111
                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())   //每次以16个数据来计算
                {
                    mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);   //低四位
                    mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);  //高四位
                    //low_mask作为掩码的索引，寻找lut_low_v的元素
                    /*
                    lut_low_v:[0,4,1,4,0,4,1,4,0,4,1,4,0,4,1,4]
                    low_mask:[15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,13]
                    low_res:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
                    low_mask对SIMILARITY_LUT进行查找
                    */
                    //SSE指令
                    mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);  
                    mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);
                    mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else if(mipp::N<uint8_t>() == 32|| mipp::N<uint8_t>() == 64)
            {       
                //128 256 512 SIMD
                CV_Assert((src.rows * src.cols) % mipp::N<uint8_t>() == 0);
                uint8_t lut_temp[mipp::N<uint8_t>()] = {0};
                for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++)
                {
                    std::copy_n(lut_low, 16, lut_temp+slice*16);
                }
                mipp::Reg<uint8_t> lut_low_v(lut_temp);
                uint8_t base_add_array[mipp::N<uint8_t>()] = {0};
                for(uint8_t slice=0; slice<mipp::N<uint8_t>(); slice+=16)
                {
                    std::copy_n(lut_low+16, 16, lut_temp+slice);
                    std::fill_n(base_add_array+slice, 16, slice);
                }
                mipp::Reg<uint8_t> base_add(base_add_array);
                mipp::Reg<uint8_t> lut_high_v(lut_temp);
                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
                {
                    mipp::Reg<uint8_t> mask_low_v((uint8_t*)lsb4_data+i);
                    mipp::Reg<uint8_t> mask_high_v((uint8_t*)msb4_data+i);
                    mask_low_v += base_add;
                    mask_high_v += base_add;
                    mipp::Reg<uint8_t> shuff_low_result = mipp::shuff(lut_low_v, mask_low_v);
                    mipp::Reg<uint8_t> shuff_high_result = mipp::shuff(lut_high_v, mask_high_v);
                    mipp::Reg<uint8_t> result = mipp::max(shuff_low_result, shuff_high_result);
                    result.store((uint8_t*)map_data + i);
                }
            }
            else
            {
                for (int i = 0; i < src.rows * src.cols; ++i)
                    map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);
            }
        }
    }
}

/**
 * \brief Convert a response map to fast linearized ordering.
 * \param[in]  response_map The 2D response map, an 8-bit image.
 * \param[out] linearized   The response map in linearized order. It has T*T rows,
 *                          each of which is a linear memory of length (W/T)*(H/T).
 * \param      T            Sampling step.
 * {
 *    a00,a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15;
 *    b00,b01,b02,b03,b04,b05,b06,b07,b08,b09,b10,b11,b12,b13,b14,b15;
 *    c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14,c15;
 *    d00,d01,d02,d03,d04,d05,d06,d07,d08,d09,d10,d11,d12,d13,d14,d15;
 *    e00,e01,e02,e03,e04,e05,e06,e07,e08,e09,e10,e11,e12,e13,e14,e15;
 *    f00,f01,f02,f03,f04,f05,f06,f07,f08,f09,f10,f11,f12,f13,f14,f15;
 *    g00,g01,g02,g03,g04,g05,g06,g07,g08,g09,g10,g11,g12,g13,g14,g15;
 *    h00,h01,h02,h03,h04,h05,h06,h07,h08,h09,h10,h11,h12,h13,h14,h15;
 * }
 * linearized[0]:a00,a04,a08,a12,e00,e04,e08,e12
 * linearized[1]:a01,a05,a09,a13,e01,e05,e09,e13
 * linearized[2]:a02,a06,a10,a14,e02,e06,e10,e14
 * linearized[3]:a03,a07,a11,a15,e03,e07,e11,e15
 * linearized[4]:b00,b04,b08,b12,f00,f04,f08,f12
 * linearized[5]:b01,b05,b09,b13,f01,f05,f09,f13
 */
static void linearize(const Mat &response_map, Mat &linearized, int T)  //T={4,8}
{
    CV_Assert(response_map.rows%T == 0);
    CV_Assert(response_map.cols%T == 0);

    // linearized has T^2 rows, where each row is a linear memory
    int mem_width = response_map.cols / T;
    int mem_height = response_map.rows / T;
    linearized.create(T*T, mem_width*mem_height,CV_8U);

    // Outer two for loops iterate over top-left T^2 starting pixels
    int index = 0;
    for (int r_start = 0; r_start < T; ++r_start)
        for (int c_start = 0; c_start < T; ++c_start)
        {
            uchar *memory = linearized.ptr(index);
            ++index;
            // Inner two loops copy every T-th pixel into the linear memory
            for (int r = r_start; r < response_map.rows; r += T)
            {
                const uchar *response_data = response_map.ptr(r);
                for (int c = c_start; c < response_map.cols; c += T) 
                            *memory++=response_data[c];
            }
        }
}
/****************************************************************************************\
*                             Linearized similarities                                                                    *
\****************************************************************************************/

static const unsigned char *accessLinearMemory(const std::vector<Mat> &linear_memories,const Feature &f, int T, int W)
{
    // W=size.width / T;
    // Retrieve the TxT grid of linear memories associated with the feature label
    const Mat &memory_grid = linear_memories[f.label];    //先找到对应的类别0～7，linear_memories为测试图片数据
    CV_DbgAssert(memory_grid.rows == T * T);
    CV_DbgAssert(f.x >= 0);    
    CV_DbgAssert(f.y >= 0);
    // The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
    /*
     * response_maps:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4;
     *                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4;
     *                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4;
     *                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4;
     *                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4;
     *                1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;
     *                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
     *                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
     * 
     * T:4进行采样
     * linearized:[4,4,4,4,4,4,4,4;
     *             4,4,4,4,4,4,4,4;
     *             4,4,4,4,4,4,4,4; 
     *             4,4,4,4,4,4,4,4;
     *             4,4,4,4,1,1,1,1;
                   4,4,4,4,1,1,1,1;
                   4,4,4,4,1,1,1,1;
                   4,4,4,4,1,1,1,1;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0;
                   4,4,4,4,0,0,0,0]
     */

    int grid_x = f.x % T;
    int grid_y = f.y % T;
    int grid_index = grid_y * T + grid_x;    //[0,T^2)
    CV_DbgAssert(grid_index >= 0);
    CV_DbgAssert(grid_index < memory_grid.rows);
    const unsigned char *memory = memory_grid.ptr(grid_index);  //哪一行
    // Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
    // input image width decimated by T. W=T^2,H=?
    int lm_x = f.x / T;
    int lm_y = f.y / T;
    int lm_index = lm_y * W + lm_x;    //W=width(测试图像宽度)/T
    CV_DbgAssert(lm_index >= 0);
    CV_DbgAssert(lm_index < memory_grid.cols);
    return memory + lm_index;   //
}

/**
 * \brief Compute similarity measure for a given template at each sampled image location.
 * Uses linear memories to compute the similarity measure as described in Fig. 7.
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.测试图像数据
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image of size (W/T, H/T).
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 */
static void similarity(const std::vector<Mat> &linear_memories, const Template &templ,Mat &dst, Size size, int T)
{
    CV_Assert(templ.features.size() < 8192);
    //Decimate input image size by factor of T
    //测试图片
    int W = size.width / T;
    int H = size.height / T;
    //Feature dimensions, decimated by factor T and rounded up
    //特征矩形框宽度
    int wf = (templ.width - 1) / T + 1; 
    int hf = (templ.height -1) / T + 1;

    //Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;
    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1;   //进行滑动距离

    dst = Mat::zeros(H, W, CV_16U);           //16-bit unsigned integers ( 0..65535 )
    short *dst_ptr = dst.ptr<short>();
    mipp::Reg<uint8_t> zero_v(uint8_t(0));    //[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    //Compute the similarity measure for this template by accumulating the contribution of each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)    //64
    {
        //Add the linear memory at the appropriate offset computed from the location of the feature in the template
        Feature f = templ.features[i];
        //Discard feature if out of bounds
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);    //测试图像
        //Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        // Process responses 16 at a time if vectorization possible
	    //mipp::N<int16_t>()：8
	    //mipp::N<uint8_t>():16
        int j = 0;
        for(; j <= template_positions -mipp::N<int16_t>()*2; j+=mipp::N<int16_t>())
        {
            //uchar转化维short，长度由16位转8位
            /*
                src8_v: [4,4,4,4,4,4,4,4,4,1,1,1,1,4,4,4]
                src16_v:[4,4,4,4,4,4,4,4]
                dst_v:  [2,2,2,2,2,2,2,2]
                res_v:  [6,6,6,6,6,6,6,6]
            */
            mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + j);
            mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
            mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + j);
            //对应位置上元素相加
            mipp::Reg<int16_t> res_v = src16_v + dst_v;
            res_v.store((int16_t*)dst_ptr + j);
        }
        for(; j<template_positions; j++) dst_ptr[j] += short(lm_ptr[j]);
    }
}
/**
 * \brief Compute similarity measure for a given template in a local region.
 *
 * \param[in]  linear_memories Vector of 8 linear memories, one for each label.
 * \param[in]  templ           Template to match against.
 * \param[out] dst             Destination 8-bit similarity image, 16x16.
 * \param      size            Size (W, H) of the original input image.
 * \param      T               Sampling step.
 * \param      center          Center of the local region.
 */
static void similarityLocal(const std::vector<Mat> &linear_memories, const Template &templ,
                            Mat &dst, Size size, int T, Point center)
{
    //This version takes a position 'center' and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 8192);
    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_16U);
  // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
  // center to get the top-left corner of the 16x16 patch.
  // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;  
    int offset_y = (center.y / T - 8) * T;
    mipp::Reg<uint8_t> zero_v = uint8_t(0);   //[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);
        // Process whole row at a time if vectorization possible
        {
            short *dst_ptr = dst.ptr<short>();
            if(mipp::N<uint8_t>() > 32)  //512 bits SIMD
            { 
                for (int row = 0; row < 16; row += mipp::N<int16_t>()/16)
                {
                    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + row*16);
                    // load lm_ptr, 16 bytes once, for half
                    uint8_t local_v[mipp::N<uint8_t>()] = {0};
                    for(int slice=0; slice<mipp::N<uint8_t>()/16/2; slice++)
                    {
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src8_v(local_v);
                    // uchar to short, once for N bytes
                    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
                    mipp::Reg<int16_t> res_v = src16_v + dst_v;
                    res_v.store((int16_t*)dst_ptr);
                    dst_ptr += mipp::N<int16_t>();
                }
            }
            else
            {   
                // 256 128 or no SIMD
            	//mipp::N<int16_t>()：8
	            //mipp::N<uint8_t>(): 16
                for (int row = 0; row < 16; ++row)
                {
                    for(int col=0; col<16; col+=mipp::N<int16_t>()) //8
                    {
                        mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr + col);
                        // uchar to short, once for N bytes
                        mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);

                        mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr + col);
                        mipp::Reg<int16_t> res_v = src16_v + dst_v;
                        res_v.store((int16_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

static void similarity_64(const std::vector<Mat> &linear_memories, const Template &templ,
                          Mat &dst, Size size, int T)
{
    // 63 features or less is a special case because the max similarity per-feature is 4.
    // 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
    // about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas a more
    // general function would use _mm_add_epi16.
    CV_Assert(templ.features.size() < 64);
    /// @todo Handle more than 255/MAX_RESPONSE features!!

    // Decimate input image size by factor of T
    int W = size.width / T;
    int H = size.height / T;

    // Feature dimensions, decimated by factor T and rounded up
    int wf = (templ.width - 1) / T + 1;
    int hf = (templ.height - 1) / T + 1;

    // Span is the range over which we can shift the template around the input image
    int span_x = W - wf;
    int span_y = H - hf;

    // Compute number of contiguous (in memory) pixels to check when sliding feature over
    // image. This allows template to wrap around left/right border incorrectly, so any
    // wrapped template matches must be filtered out!
    int template_positions = span_y * W + span_x + 1; // why add 1?
    //int template_positions = (span_y - 1) * W + span_x; // More correct?

    /// @todo In old code, dst is buffer of size m_U. Could make it something like
    /// (span_x)x(span_y) instead?
    dst = Mat::zeros(H, W, CV_8U);
    uchar *dst_ptr = dst.ptr<uchar>();

    // Compute the similarity measure for this template by accumulating the contribution of
    // each feature
    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        // Add the linear memory at the appropriate offset computed from the location of
        // the feature in the template
        Feature f = templ.features[i];
        // Discard feature if out of bounds
        /// @todo Shouldn't actually see x or y < 0 here?
        if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
            continue;
        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        // Now we do an aligned/unaligned add of dst_ptr and lm_ptr with template_positions elements
        int j = 0;

        for(; j <= template_positions -mipp::N<uint8_t>(); j+=mipp::N<uint8_t>()){
            mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + j);
            mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + j);

            mipp::Reg<uint8_t> res_v = src_v + dst_v;
            res_v.store((uint8_t*)dst_ptr + j);
        }

        for(; j<template_positions; j++)
            dst_ptr[j] += lm_ptr[j];
    }
}

static void similarityLocal_64(const std::vector<Mat> &linear_memories, const Template &templ,
                               Mat &dst, Size size, int T, Point center)
{
    // Similar to whole-image similarity() above. This version takes a position 'center'
    // and computes the energy in the 16x16 patch centered on it.
    CV_Assert(templ.features.size() < 64);

    // Compute the similarity map in a 16x16 patch around center
    int W = size.width / T;
    dst = Mat::zeros(16, 16, CV_8U);

    // Offset each feature point by the requested center. Further adjust to (-8,-8) from the
    // center to get the top-left corner of the 16x16 patch.
    // NOTE: We make the offsets multiples of T to agree with results of the original code.
    int offset_x = (center.x / T - 8) * T;
    int offset_y = (center.y / T - 8) * T;

    for (int i = 0; i < (int)templ.features.size(); ++i)
    {
        Feature f = templ.features[i];
        f.x += offset_x;
        f.y += offset_y;
        // Discard feature if out of bounds, possibly due to applying the offset
        if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
            continue;

        const uchar *lm_ptr = accessLinearMemory(linear_memories, f, T, W);

        {
            uchar *dst_ptr = dst.ptr<uchar>();

            if(mipp::N<uint8_t>() > 16)
            { // 256 or 512 bits SIMD
                for (int row = 0; row < 16; row += mipp::N<uint8_t>()/16)
                {
                    mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr);
                    // load lm_ptr, 16 bytes once
                    uint8_t local_v[mipp::N<uint8_t>()];
                    for(int slice=0; slice<mipp::N<uint8_t>()/16; slice++){
                        std::copy_n(lm_ptr, 16, &local_v[16*slice]);
                        lm_ptr += W;
                    }
                    mipp::Reg<uint8_t> src_v(local_v);

                    mipp::Reg<uint8_t> res_v = src_v + dst_v;
                    res_v.store((uint8_t*)dst_ptr);

                    dst_ptr += mipp::N<uint8_t>();
                }
            }else{ // 128 or no SIMD
                for (int row = 0; row < 16; ++row){
                    for(int col=0; col<16; col+=mipp::N<uint8_t>()){
                        mipp::Reg<uint8_t> src_v((uint8_t*)lm_ptr + col);
                        mipp::Reg<uint8_t> dst_v((uint8_t*)dst_ptr + col);
                        mipp::Reg<uint8_t> res_v = src_v + dst_v;
                        res_v.store((uint8_t*)dst_ptr + col);
                    }
                    dst_ptr += 16;
                    lm_ptr += W;
                }
            }
        }
    }
}

/****************************************************************************************\
*                                                             High-level Detector API                                                                    *
\****************************************************************************************/

Detector::Detector()
{
    // this->modality = makePtr<ColorGradient>();
    // pyramid_levels = 2;
    // T_at_level.push_back(4);
    // T_at_level.push_back(8);
}

Detector::Detector(std::vector<int> T)
{
    this->modality = makePtr<ColorGradient>();
    pyramid_levels = T.size();
    T_at_level = T;
}

Detector::Detector(int num_features, std::vector<int> T, float weak_thresh, float strong_threash)
{
    this->modality = makePtr<ColorGradient>(weak_thresh, num_features, strong_threash);
    pyramid_levels = T.size();
    T_at_level = T;
}

std::vector<Match> Detector::match(Mat source_, int threshold,const std::vector<std::string> &class_ids,const Mat mask) const
{
    std::vector<Match> matches;
    std::vector<Ptr<ColorGradientPyramid>> quantizers;
    CV_Assert(mask.empty() || mask.size() == source_.size());
    quantizers.push_back(modality->process(source_, mask));
    //Indexed as [pyramid level][ColorGradient][quantization]
    LinearMemoryPyramid lm_pyramid(pyramid_levels,std::vector<LinearMemories>(1, LinearMemories(8)));
    std::vector<Size> sizes;
    
    //建立ResponseMap,T_at_level={4,8}
    for (int l = 0; l < pyramid_levels; ++l)
    {
        int T = T_at_level[l];
        std::vector<LinearMemories> &lm_level = lm_pyramid[l];
        if (l > 0)
        {
            for(int i = 0; i < (int)quantizers.size(); ++i) 
                    quantizers[i]->pyrDown();
        }
        Mat quantized, spread_quantized;
        std::vector<Mat> response_maps;                    //8个响应图
        for (int i = 0; i < (int)quantizers.size(); ++i)   //quantizers.size()=1
        { 
            quantizers[i]->quantize(quantized); //quantized:1,2,4,8,16,32,64,128,quantized是单通道矩阵
            spread(quantized, spread_quantized, T);
            computeResponseMaps(spread_quantized,response_maps);
            LinearMemories &memories = lm_level[i];
            for(int j = 0; j < 8; ++j) 
                    linearize(response_maps[j], memories[j], T);
        }
        sizes.push_back(quantized.size());
    }

    //特征图与模板图之间对比
    if (class_ids.empty())
    {
        TemplatesMap::const_iterator it = class_templates.begin(), itend = class_templates.end();
        for (; it != itend; ++it) 
            matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
    }
    else
    {
        for (int i = 0; i < (int)class_ids.size(); ++i)
        {
            TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
            if (it != class_templates.end())
                matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
        }
    }
    // Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
    std::sort(matches.begin(), matches.end());
    std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
    matches.erase(new_end, matches.end());
    return matches;
}

struct MatchPredicate
{
    MatchPredicate(float _threshold) : threshold(_threshold) {}
    bool operator()(const Match &m) { return m.similarity < threshold; }
    float threshold;
};

void Detector::matchClass(const LinearMemoryPyramid &lm_pyramid,
                          const std::vector<Size> &sizes,int threshold, 
                          std::vector<Match> &matches,const std::string &class_id,
                          const std::vector<TemplatePyramid> &template_pyramids) const
{
    // cout<<"template size:"<<template_pyramids.size()<<endl;       //11
    // cout<<"LinearMemoryPyramid size:"<<lm_pyramid.size()<<endl;   //2
    // cout<<"Color gradent:"<<lm_pyramid[0].size()<<endl;           //1
    // cout<<"quant labels:"<<lm_pyramid[0][0].size()<<endl;         //8
    #pragma omp declare reduction(omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
    #pragma omp parallel for reduction(omp_insert:matches)
    for (size_t template_id = 0; template_id < template_pyramids.size(); ++template_id)  //10
    {   
        const TemplatePyramid &tp = template_pyramids[template_id];
        const std::vector<LinearMemories> &lowest_lm = lm_pyramid.back();//从T=8开始拿数据
        std::vector<Match> candidates;
        {
            Mat similarities;
            int lowest_start = static_cast<int>(tp.size() - 1);  //1
            int lowest_T = T_at_level.back();                    //8
            int num_features = 0;
            {
                const Template &templ = tp[lowest_start];        //第1金字塔层所有模板(包括angle与scale的不同)
                num_features += static_cast<int>(templ.features.size());
                if (templ.features.size() < 64)
                {
                    similarity_64(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                    similarities.convertTo(similarities, CV_16U);
                }
                else if (templ.features.size() < 8192) 
                {
                    //lowest_lm[0]：T的采样点(T*T,mem*mem),size:8
                    //templ：每层金字塔层所有angle，scale的模板
                    similarity(lowest_lm[0], templ, similarities, sizes.back(), lowest_T);
                }
                else
                {
                    CV_Error(Error::StsBadArg, "feature size too large");
                }
            }
            /*
                similarities=(H,W,CV_16U)
                W=size.width / T
                H=size.height/ T
            */
            for (int r = 0; r < similarities.rows; ++r)
            {
                ushort *row = similarities.ptr<ushort>(r);
                for (int c = 0; c < similarities.cols; ++c)
                {
                    int raw_score = row[c];
                    float score = (raw_score*100.f) / (4 * num_features);   //最大元素4*num_features
                    if (score >=threshold)
                    {
                        int offset = lowest_T / 2 + (lowest_T % 2 - 1);     //中心点3
                        int x = c * lowest_T + offset;                      //lowest_T=T=8
                        int y = r * lowest_T + offset;
                        candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
                    }
                }
            }
        }
        //Locally refine each match by marching up the pyramid
        for (int l = pyramid_levels - 2; l >= 0; --l) //pyramid_levels:2
        {
            const std::vector<LinearMemories> &lms = lm_pyramid[l]; //金字塔0层时数据
            int T = T_at_level[l];              //4
            int start = static_cast<int>(l);
            Size size = sizes[l];               //输入图像
            int border = 8 * T;                 //32,在测试图片中padding
            int offset = T / 2 + (T % 2 - 1);   //3
            int max_x = size.width - tp[start].width- border;   //tp是特征矩形框宽和高
            int max_y = size.height - tp[start].height- border; 

            Mat similarities2;
            for (int m = 0; m < (int)candidates.size(); ++m)
            {
                Match &match2 = candidates[m];
                int x = match2.x * 2 + 1; 
                int y = match2.y * 2 + 1;

                // Require 8 (reduced) row/cols to the up/left
                x = std::max(x, border);
                y = std::max(y, border);

                // Require 8 (reduced) row/cols to the down/left, plus the template size
                x = std::min(x, max_x);
                y = std::min(y, max_y);

                // Compute local similarity maps for each modality
                int numFeatures = 0;
                {
                    const Template &templ = tp[start];
                    numFeatures += static_cast<int>(templ.features.size());
                    if (templ.features.size() < 64)
                    {
                        similarityLocal_64(lms[0], templ, similarities2, size, T, Point(x, y));
                        similarities2.convertTo(similarities2, CV_16U);
                    }
                    else if (templ.features.size() < 8192)  //128
                    {
                        similarityLocal(lms[0], templ, similarities2, size, T, Point(x, y));
                    }
                    else
                    {
                        CV_Error(Error::StsBadArg, "feature size too large");
                    }
                }

                // Find best local adjustment
                float best_score = 0;
                int best_r = -1, best_c = -1;
                for (int r = 0; r < similarities2.rows; ++r)
                {
                    ushort *row = similarities2.ptr<ushort>(r);
                    for (int c = 0; c < similarities2.cols; ++c)
                    {
                        int score_int = row[c];
                        float score = (score_int*100.f) / (4 * numFeatures); //* 100.f
                        if (score > best_score)
                        {
                            best_score = score;
                            best_r = r;
                            best_c = c;
                        }
                    }
                }
                // Update current match
                match2.similarity = best_score;
                match2.x = (x / T - 8 + best_c) * T + offset;
                match2.y = (y / T - 8 + best_r) * T + offset;
            }
            // Filter out any matches that drop below the similarity threshold
            std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),MatchPredicate(threshold));
            candidates.erase(new_end, candidates.end());
        }
        matches.insert(matches.end(), candidates.begin(), candidates.end());
    }
}

int Detector::addTemplate(const Mat source, const std::string &class_id,const Mat &object_mask, int num_features)
{
    //std::vector<std::vector<Template>>
    std::vector<TemplatePyramid> &template_pyramids = class_templates[class_id];
    int template_id = static_cast<int>(template_pyramids.size());
    TemplatePyramid tp;
    tp.resize(pyramid_levels);    //pyramid_levels:金字塔层数2
    {
        // Extract a template at each pyramid level
        Ptr<ColorGradientPyramid> qp = modality->process(source, object_mask);
        if(num_features > 0) qp->num_features = num_features;   //新增加特征点数目
        for (int l = 0; l < pyramid_levels; ++l)
        {
            if (l > 0)  qp->pyrDown();
            bool success = qp->extractTemplate(tp[l]);
            if (!success) return -1;
        }
    }
    cropTemplates(tp,source);
    template_pyramids.push_back(tp);
    return template_id;
}
const std::vector<Template> &Detector::getTemplates(const std::string &class_id, int template_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    CV_Assert(i != class_templates.end());
    CV_Assert(i->second.size() > size_t(template_id));
    return i->second[template_id];
}

int Detector::numTemplates() const
{
    int ret = 0;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
        ret += static_cast<int>(i->second.size());
    return ret;
}

int Detector::numTemplates(const std::string &class_id) const
{
    TemplatesMap::const_iterator i = class_templates.find(class_id);
    if (i == class_templates.end())
        return 0;
    return static_cast<int>(i->second.size());
}

std::vector<std::string> Detector::classIds() const
{
    std::vector<std::string> ids;
    TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
    for (; i != iend; ++i)
    {
        ids.push_back(i->first);
    }

    return ids;
}

void Detector::read(const FileNode &fn)
{
    class_templates.clear();
    pyramid_levels = fn["pyramid_levels"];
    fn["T"] >> T_at_level;

    modality = makePtr<ColorGradient>();
}

void Detector::write(FileStorage &fs) const
{
    fs << "pyramid_levels" << pyramid_levels;
    fs << "T" << T_at_level;

    modality->write(fs);
}

std::string Detector::readClass(const FileNode &fn, const std::string &class_id_override)
{
    // Detector should not already have this class
    String class_id;
    if (class_id_override.empty())
    {
        String class_id_tmp = fn["class_id"];
        CV_Assert(class_templates.find(class_id_tmp) == class_templates.end());
        class_id = class_id_tmp;
    }
    else
    {
        class_id = class_id_override;
    }

    TemplatesMap::value_type v(class_id, std::vector<TemplatePyramid>());
    std::vector<TemplatePyramid> &tps = v.second;
    int expected_id = 0;

    FileNode tps_fn = fn["template_pyramids"];
    tps.resize(tps_fn.size());
    FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
    for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
    {
        int template_id = (*tps_it)["template_id"];
        CV_Assert(template_id == expected_id);
        FileNode templates_fn = (*tps_it)["templates"];
        tps[template_id].resize(templates_fn.size());

        FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
        int idx = 0;
        for (; templ_it != templ_it_end; ++templ_it)
        {
            tps[template_id][idx++].read(*templ_it);
        }
    }
    class_templates.insert(v);
    return class_id;
}

void Detector::writeClass(const std::string &class_id, FileStorage &fs) const
{
    TemplatesMap::const_iterator it = class_templates.find(class_id);
    CV_Assert(it != class_templates.end());
    const std::vector<TemplatePyramid> &tps = it->second;

    fs << "class_id" << it->first;
    fs << "pyramid_levels" << pyramid_levels;
    fs << "template_pyramids"
       << "[";
    for (size_t i = 0; i < tps.size(); ++i)
    {
        const TemplatePyramid &tp = tps[i];
        fs << "{";
        fs << "template_id" << int(i); //TODO is this cast correct? won't be good if rolls over...
        fs << "templates"
           << "[";
        for (size_t j = 0; j < tp.size(); ++j)
        {
            fs << "{";
            tp[j].write(fs);
            fs << "}"; // current template
        }
        fs << "]"; // templates
        fs << "}"; // current pyramid
    }
    fs << "]"; // pyramids
}

void Detector::readClasses(const std::vector<std::string> &class_ids,const std::string &format)
{
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        const String &class_id = class_ids[i];
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::READ);
        readClass(fs.root());
    }
}

void Detector::writeClasses(const std::string &format) const
{
    TemplatesMap::const_iterator it = class_templates.begin(), it_end = class_templates.end();
    for (; it != it_end; ++it)
    {
        const String &class_id = it->first;
        String filename = cv::format(format.c_str(), class_id.c_str());
        FileStorage fs(filename, FileStorage::WRITE);
        writeClass(class_id, fs);
    }
}

//重复训练时的清除
void Detector::deleteClass(const std::string class_id)
{
    int num=numTemplates(class_id);
    if(!num) return;
    std::vector<TemplatePyramid> &template_pyramids=class_templates[class_id];
    for(size_t i = 0; i < template_pyramids.size(); i++)
    {
        std::vector<Template>& temp=template_pyramids[i];
        for(size_t j=0;j<temp.size();j++)
        {
            temp[j].features.clear();
        }
        temp.clear();
    }
    template_pyramids.clear();
    TemplatesMap::iterator iterator = class_templates.find(class_id);
    class_templates.erase(iterator);
};

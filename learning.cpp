#include "latou.h"
void orUnaligned8u(const uchar *src, const int src_stride,uchar *dst, 
                   const int dst_stride,const int width, const int height)
{
    for (int r = 0; r < height; ++r)
    {
        int c = 0;
        //16位进行对齐,T为4,c分别为0,15,14,13,移动多少位可以对齐,src偏移c位可以对齐
        while (reinterpret_cast<unsigned long long>(src + c) % 16 != 0) 
        {
            dst[c]|= src[c];  
            c++;
        }
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

void spread(const Mat &src, Mat &dst, int T)
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
 */
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = {0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 1, 1, 4, 4, 0, 1, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4};
static void computeResponseMaps(const Mat &src, std::vector<Mat> &response_maps)
{
    CV_Assert((src.rows * src.cols) % 16 == 0);
    response_maps.resize(8);
    for (int i = 0; i < 8; ++i)
        response_maps[i].create(src.size(), CV_8U);

    Mat lsb4(src.size(), CV_8U);
    Mat msb4(src.size(), CV_8U);

    for (int r = 0; r < src.rows; ++r)
    {
        const uchar *src_r = src.ptr(r);
        uchar *lsb4_r = lsb4.ptr(r);
        uchar *msb4_r = msb4.ptr(r);
        for (int c = 0; c < src.cols; ++c)
        {
            lsb4_r[c] = src_r[c] & 15;           //8位二进制数低四位
            msb4_r[c] = (src_r[c] & 240) >> 4;   //8位二进制数高四位
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
                mipp::Reg<uint8_t> lut_low_v((uint8_t*)lut_low);       //16位  lsb
                mipp::Reg<uint8_t> lut_high_v((uint8_t*)lut_low + 16); //
                for (int i = 0; i < src.rows * src.cols; i += mipp::N<uint8_t>())
                {
                    mipp::Reg<uint8_t> low_mask((uint8_t*)lsb4_data + i);   //每位数据的低8位
                    mipp::Reg<uint8_t> high_mask((uint8_t*)msb4_data + i);  //每位数据的高8位
                    //low_mask作为掩码的索引，寻找lut_low_v的元素
                    /*
                    lut_low_v:[0,4,1,4,0,4,1,4,0,4,1,4,0,4,1,4]
                    low_mask:[15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,13]
                    low_res:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
                    */
                    mipp::Reg<uint8_t> low_res = mipp::shuff(lut_low_v, low_mask);  
                    mipp::Reg<uint8_t> high_res = mipp::shuff(lut_high_v, high_mask);
                    mipp::Reg<uint8_t> result = mipp::max(low_res, high_res);
                    
                    // cout<<"lut_low_v:"<<lut_low_v<<endl;
                    // cout<<"low_mask:"<<low_mask<<endl;
                    // cout<<"low_res:"<<low_res<<endl;

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





void test()
{
    //1,2,4,8,16,32,64,128
    uchar data[128]={
                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                    2,2,2,2,2,2,2,2,32,2,2,2,2,2,2,1,
                    4,4,4,4,64,4,4,4,4,4,4,4,4,4,4,4,
                    8,8,8,8,8,128,8,16,8,8,8,8,8,8,8,8,
                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                    8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
                    };
    Mat src=Mat(8,16,CV_8UC1,data);
    std::cout<<src<<std::endl;
    Mat dst;
    std::vector<Mat> response_maps;
    spread(src,dst,4);
    computeResponseMaps(dst,response_maps);
    Mat linearized;
    linearize(response_maps[0],linearized,4);
    //cout<<"response_maps:"<<response_maps[0]<<endl;
    cout<<"linearized:"<<linearized<<endl;
    //cout<<"linearized_rows:"<<linearized.rows<<",cols:"<<linearized.cols<<endl;

    dst = Mat::ones(linearized.rows, linearized.cols, CV_16U)*2;
    short *dst_ptr = dst.ptr<short>();
    mipp::Reg<uint8_t> zero_v(uint8_t(0));

    const uchar* lm_ptr=&(linearized.ptr<uchar>(3)[3]);
    mipp::Reg<uint8_t> src8_v((uint8_t*)lm_ptr);
    cout<<src8_v<<endl;
    mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r); 
    cout<<src16_v<<endl;
    mipp::Reg<int16_t> dst_v((int16_t*)dst_ptr);
    mipp::Reg<int16_t> res_v = src16_v + dst_v;
    cout<<res_v<<endl;


}

int main(int argc,char **argv)
{
    test();
}
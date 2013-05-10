#ifndef __CUDA_H264_INC__
#define __CUDA_H264_INC__



#define imgpel unsigned short
#define pel_t imgpel

#ifdef __CUDACC__
extern "C"
{
#endif
#include "global.h"
void cuda_validate_arguments();
void cuda_copy_one_frame_and_bind_texture(imgpel* imgY,int width, int height);
void cuda_init_motion_search_module();
void cuda_free();
void cuda_begin_encode_frame();
void cuda_free_device_imgY(cudaArray_t arr);
void cuda_alloc_device_imgY(cudaArray_t* arr);
void cuda_end_encode_frame();
int                                                   //  ==> minimum motion cost after search
cudaFastFullPelBlockMotionSearch(pel_t**   orig_pic,     // <--  not used
                              short     ref,          // <--  reference frame (0... or -1 (backward))
                              int       list,
                              int       pic_pix_x,    // <--  absolute x-coordinate of regarded AxB block
                              int       pic_pix_y,    // <--  absolute y-coordinate of regarded AxB block
                              int       blocktype,    // <--  block type (1-16x16 ... 7-4x4)
                              short     pred_mv_x,    // <--  motion vector predictor (x) in sub-pel units
                              short     pred_mv_y,    // <--  motion vector predictor (y) in sub-pel units
                              short*    mv_x,         //  --> motion vector (x) - in pel units
                              short*    mv_y,         //  --> motion vector (y) - in pel units
                              int       search_range, // <--  1-d search range in pel units
                              int       min_mcost,    // <--  minimum motion cost (cost for center or huge value)
                              int       lambda_factor);      // <--  lagrangian parameter for determining motion cost
#ifdef __CUDACC__
}
#endif


#endif //__CUDA_H264_INC__

#ifndef __CUDA_H264_INC__
#define __CUDA_H264_INC__


#include "global.h"


extern "C" void quarter_filter(imgpel** imgY, int width, int height, imgpel** outY4);

#endif //__CUDA_H264_INC__

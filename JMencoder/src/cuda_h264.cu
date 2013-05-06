#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>
#include "cuda_h264.h"
extern "C"{
#include "defines.h"
}
extern "C"{
#include "mbuffer.h"
}

static void cuda_h264_error(cudaError_t err, const char* do_what, const char* file, int line)
{
	fprintf(stderr, "in %s at %d line:when %s .For %s\n", file, line, do_what, cudaGetErrorString(err));
	flush_dpb();
	exit(EXIT_FAILURE);
}

#define CUDA_ERROR(err, do_what) cuda_h264_error(err, do_what, __FILE__, __LINE__)
#define CUDA_CHECK(do_what, x) do{ cudaError_t err; if((err = (x)) != cudaSuccess) CUDA_ERROR(err,do_what); }while(0)



//#define IMG_PAD_SIZE 4

//#define imgpel unsigned short


//#define Clip3(a,b,x) (x>a?(x<b?x:b):a)
//#define min(a,b) (a>b?b:a)
//#define BLOCK_SIZE 8


//thread 9x4 in one MB 
//16x16 in 1/4 pel  -> one Block

texture<imgpel, cudaTextureType2D>  tex_ref_imgY_src;

__global__ void half_pel_filter(imgpel* d_outY4, int outY4_pitch)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int mappedty = ty + 2;
	int xin1_4 = blockIdx.x << 4 + tx << 2;
	int yin1_4 = blockIdx.y << 4 + ty << 2;
	int getX = (xin1_4 - 16)/4;//可能存在负数，不能随便使用移位
	int getY = (yin1_4 - 16 - 2 ) /4;
	__shared__ int hpel_temp[9][8];//临时的1/2差值后的点与整数像素点

	int pel = tex2D(tex_ref_imgY_src, getX, getY);
	hpel_temp[ty][tx<<1] = pel << 10;
	pel = 
		20 * (pel + tex2D(tex_ref_imgY_src, getX+1, getY))
		-5 * (tex2D(tex_ref_imgY_src, getX-1, getY) + tex2D(tex_ref_imgY_src, getX+2, getY))
		+ (tex2D(tex_ref_imgY_src, getX-2, getY) + tex2D(tex_ref_imgY_src, getX + 3, getY));

	hpel_temp[threadIdx.y][threadIdx.x<<1 + 1] = pel << 5;//1024倍
	__syncthreads();//栅栏同步

	if(ty < 4)
	{
		d_outY4[yin1_4 * outY4_pitch + xin1_4] = (imgpel)(hpel_temp[mappedty][tx<<1] >> 10);//整数点
		d_outY4[yin1_4 * outY4_pitch + xin1_4 + 2] = (imgpel)Clip3(0, 255, (hpel_temp[mappedty][tx<<1+1] + 512) >> 10);//整数行水平1/2插值点

		tx <<= 1;
		for(int i = 0; i < 2; ++i)
		{
			tx += i;
			pel = 
				Clip3(0, 255, 
						(20 * (hpel_temp[mappedty][tx] + hpel_temp[mappedty+1][tx])
				-5 * (hpel_temp[mappedty-1][tx] + hpel_temp[mappedty+2][tx])
				+ (hpel_temp[mappedty-2][tx] + hpel_temp[mappedty+3][tx]) + 512) >> 10
					 );
			
			d_outY4[(yin1_4 + 2) * outY4_pitch + xin1_4 + i<<1] = (imgpel)pel;//垂直1/2差值, 原来的32倍
		}
	}
}

//thread 8x8 in one MB
//one 16x16 in 1/4 -> one Block
__global__ void quarter_pel_filter(imgpel* d_outY4, int outY4_pitch, int padded4width, int padded4height)
{
	__shared__ int temp[9][9];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int xin1_4 = blockIdx.x << 4 + tx << 1;
	int yin1_4 = blockIdx.y << 4 + ty << 1;
	temp[ty][tx] = d_outY4[yin1_4 * outY4_pitch + xin1_4]; //整数像素
	if(tx == 7)
		temp[ty][8] = d_outY4[yin1_4 * outY4_pitch+ min(padded4width - 2, xin1_4 + 2)];
	if(ty == 7)
		temp[8][tx] = d_outY4[min(padded4height - 2,yin1_4 + 2) * outY4_pitch + xin1_4];

	__syncthreads();//栅栏同步

	int h_qpel = temp[ty][tx];//水平1/4像素
	h_qpel = (h_qpel + temp[ty][tx+1] + 1) >> 1;
	int v_qpel = temp[ty][tx];//垂直1/4像素
	v_qpel = (v_qpel + temp[ty+1][tx] + 1) >> 1;

	int i_qpel;//倾斜1/4像素
	int off = (tx%2== 0 && ty%2==0);
	i_qpel = (temp[ty+off][tx] + temp[ty+1-off][tx+1] + 1) >> 1;
	
	d_outY4[yin1_4 * outY4_pitch+ xin1_4] = (imgpel)temp[ty][tx];//整数与1/2像素
	d_outY4[yin1_4 * outY4_pitch+ xin1_4 + 1] = (imgpel)h_qpel;//1/4水平
	d_outY4[(yin1_4+1) * outY4_pitch + xin1_4] = (imgpel)v_qpel;//1/4垂直
	d_outY4[(yin1_4+1) * outY4_pitch + xin1_4 + 1] = (imgpel)i_qpel;//1/4 倾斜
}


extern "C" 
void quarter_filter(imgpel** imgY, int width, int height, imgpel** outY4)
{

	
	imgpel* d_outY4;
	size_t d_outY4_pitch;
	int padded4width = (width + IMG_PAD_SIZE * 2) * 4;
	int padded4height = (height + IMG_PAD_SIZE * 2) * 4;
	int padded4width_in_bytes = padded4width * sizeof(imgpel);

	CUDA_CHECK("alloc memory for devie imgY in 1/4",
			cudaMallocPitch(&d_outY4, &d_outY4_pitch, padded4width_in_bytes, padded4height)
			);

	//get device imgY src
	//imgpel* d_imgY_src;
	cudaArray_t d_imgY_src;
	cudaChannelFormatDesc imgY_src_format = cudaCreateChannelDesc<imgpel>();
	CUDA_CHECK("alloc memory for device imgY src", 
			cudaMallocArray(&d_imgY_src, &imgY_src_format,width,height, cudaArrayTextureGather));
	int width_in_bytes = width*sizeof(imgpel);
	int src_pitch = width_in_bytes;
	CUDA_CHECK("copy imgY to devic imgY",
			cudaMemcpy2DToArray(d_imgY_src, 0, 0, imgY[0], src_pitch ,width_in_bytes, height, cudaMemcpyHostToDevice)
			);

	//绑定纹理
	tex_ref_imgY_src.addressMode[0] = cudaAddressModeClamp;//钳位访问
	tex_ref_imgY_src.normalized = false;
	CUDA_CHECK("bind imgY src to tex_ref_imgY_src",
			cudaBindTextureToArray(&tex_ref_imgY_src, d_imgY_src, & tex_ref_imgY_src.channelDesc)
			);
	

	//开始滤波
	dim3 helf_pel_block_dim(4,10);//x=4 y = 10
	dim3 quarter_pel_block_dim(8,8);
	dim3 dimGrid(padded4width/16, padded4height/16);
	half_pel_filter<<<helf_pel_block_dim, dimGrid>>>(d_outY4, d_outY4_pitch);
	quarter_pel_filter<<<quarter_pel_block_dim, dimGrid>>>(d_outY4, d_outY4_pitch, padded4width, padded4height);

	size_t host_ouY4_pitch_in_bytes = padded4width_in_bytes;
	CUDA_CHECK("copy img with padded border in 1/4 back to host",
			cudaMemcpy2D(outY4[0],host_ouY4_pitch_in_bytes,d_outY4,d_outY4_pitch, padded4width_in_bytes,padded4height,cudaMemcpyDeviceToHost)
			);


	//TODO free????
}


extern unsigned int* byte_abs;//外部绝对值表
extern int byte_abs_range;//外部绝对值表长度
extern int* mvbits;//外部mvbit表 
extern int* refbits;//外部refbit表
texture<unsigned int, cudaTextureType1D> texref_byte_abs;//绝对值表纹理
texture<int,cudaTextureType1D> tex_ref_mvbits;//mvbits表纹理
texture<int,cudaTextureType1D> tex_ref_refbits;//refbits表纹理

static unsigned int* g_byte_abs_on_device;//设备绝对值表
static int * g_mvbits_on_device;//设备mvbits表
static int * g_refbits_on_device;//设备refbits表
//this function mush be called after init_img init_motion_search



extern "C" void cuda_init()
{
	cudaError_t err;
	unsigned int * byte_abs_start = byte_abs - byte_abs_range/2;
	cudaChannelFormatDesc byte_abs_format = 
		cudaCreateChannelDesc(sizeof(*byte_abs), 0, 0, 0, cudaChannelFormatKindUnsigned);
	size_t size_byte_abs = byte_abs_range * sizeof(*byte_abs);
	size_t size_mvbits;
	size_t size_refbits;

	//byte_abs_texture_ref.addressMode[0] = cudaAddressModeClamp;//钳位寻址模式,超出寻址范围钳住最大或者最小！

	if((err = cudaMalloc(&g_byte_abs_on_device, size_byte_abs)) != cudaSuccess)
		CUDA_ERROR(err, "alloc memory for device byte_abs");
	//对于使用cudaMalloc获得device内存，第一遍了offset一定返回0
	//cudaMemcpy(g_byte_abs_on_device, byte_abs_start, 

	//使用tex1Dfetch(ref, x)摘取纹理
	if((err = cudaBindTexture(NULL, &texref_byte_abs, g_byte_abs_on_device, &byte_abs_format, size_byte_abs)) != cudaSuccess)
		CUDA_ERROR(err, "bind texture for byte_abs");
		
}

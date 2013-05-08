#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<math.h>
#include <cuda_runtime.h>
#include "cuda_h264.h"
extern "C"
{
#include "defines.h"
#include "mbuffer.h"
}



#define ThreadPerBlock 512


#define BlockOffset4x4 0
#define BLOCK_TYPE_SAD_COUNT (16+8+8+4+2+2+1) //16_4x4+8_8x4+8_4x8+4_8x8+2_16x8+2_8x16+1_16x16
#define CUDA_ERROR(err, do_what) cuda_h264_error(err, do_what, __FILE__, __LINE__)
#define CUDA_CHECK(do_what, x) do{ cudaError_t err; if((err = (x)) != cudaSuccess) CUDA_ERROR(err,do_what); }while(0) 
#define CUDA_2D_Element(type, base_addr, x, y, pitch) \
	((type*)((char*)base_addr + (y)*(pitch)))[x]
#define CUDA_3D_Element(type, pitchedptr, x, y, z) \
	((type*)((char*)pitchedptr.ptr + pitchedptr.pitch * pitchedptr.ysize * (z) + (y) * pitchedptr.pitch))[x]








//外部表from JM工程
extern unsigned int* byte_abs;//外部绝对值表
extern int byte_abs_range;//外部绝对值表长度
extern int* mvbits;//外部mvbit表 
extern int* refbits;//外部refbit表
extern short* spiral_search_x;
extern short* spiral_search_y;


//[ref][3D ]
//ref维位于host
//后面三维位于device
//depth-> mb (z)
//width-> pos_of_mv (x)
//each AxB block index->height (y)
cudaPitchedPtr* g_blockSAD;
texture<unsigned int, cudaTextureType1D> g_tex_ref_byte_abs;//绝对值表纹理
texture<int,cudaTextureType1D> g_tex_ref_mvbits;//mvbits表纹理
texture<int,cudaTextureType1D> g_tex_ref_refbits;//refbits表纹理
texture<imgpel, 2> g_tex_ref_img_ref;
texture<imgpel, 2> g_tex_ref_img_cur;
texture<short, 1> g_tex_ref_spiral_search_x;
texture<short, 1> g_tex_ref_spiral_search_y;
texture<short, 1> g_tex_ref_mb_search_center_x;
texture<short, 1> g_tex_ref_mb_search_center_y;
texture<imgpel, 2> g_tex_ref_imgY_src;
texture<imgpel, 2> g_tex_ref_imgY_ref;

__constant__ int gd_byte_abs_offset;
__constant__ int gd_mvbits_offset;



__constant__ int gd_max_search_pos_nr;
__constant__ int gd_max_macroblock_nr;
__constant__ int gd_picture_width_in_pel;
__constant__ int gd_picture_width_in_mb;
__constant__ int gd_picture_height_in_pel;
__constant__ int gd_picture_height_in_mb;
__constant__ int gd_max_threadx_nr_requried;




static unsigned int* gd_byte_abs;//设备绝对值表
static int * gd_mvbits;//设备mvbits表
static int * gd_refbits;//设备refbits表
static short* gd_spiral_search_x;
static short* gd_spiral_search_y;
extern void cuda_h264_error(cudaError_t err, const char* do_what, const char* file, int line) 
{
	fprintf(stderr, "in %s at %d line:when %s .For %s\n", file, line, do_what, cudaGetErrorString(err));
	flush_dpb();
	exit(EXIT_FAILURE);
}

void cuda_init_blockSAD(int max_num_references, int max_search_points, int img_width, int img_height)
{
	int mb_count = (img_width/16) * (img_height / 16);
	int width_in_bytes = max_search_points*sizeof(int);
	struct cudaExtent block3DExtent = make_cudaExtent(width_in_bytes,BLOCK_TYPE_SAD_COUNT,mb_count);

	if((g_blockSAD = (cudaPitchedPtr*)calloc(sizeof(cudaPitchedPtr), max_num_references)) == NULL)
		no_mem_exit("alloc cudaPitchPtr in host");

	for(int i = 0; i < max_num_references; ++i)
	{
		CUDA_CHECK("alloc 3d blockSAD in device",
				cudaMalloc3D(&g_blockSAD[i],block3DExtent));
	}
}

//初始化cuda motion search
//call after Init_Motion_Search_Module ()
extern "C" void init_cuda_motion_search_module()
{
	int search_range               = input->search_range;
	int max_search_points          = max(9, (2*search_range+1)*(2*search_range+1));
	//int max_ref_bits               = 1 + 2 * (int)floor(log(max(16,img->max_num_references+1)) / log(2) + 1e-10);
	//int max_ref                    = (1<<((max_ref_bits>>1)+1))-1;
	int number_of_subpel_positions = 4 * (2*search_range+3);
	int max_mv_bits                = 3 + 2 * (int)ceil (log(number_of_subpel_positions+1) / log(2) + 1e-10);
	int max_mvd                    = (1<<( max_mv_bits >>1))-1;

	size_t spiral_search_size = max_search_points*sizeof(short);
	size_t mvbits_size = (2*max_mvd+1)*sizeof(int);
	//size_t refbits_size = max_ref*sizeof(int);
	size_t byte_abs_size = byte_abs_range*sizeof(unsigned int);

	//create device array
	CUDA_CHECK("alloc device spiral_search_x",cudaMalloc(&gd_spiral_search_x, spiral_search_size));
	CUDA_CHECK("alloc device spiral_search_y",cudaMalloc(&gd_spiral_search_y, spiral_search_size));
	CUDA_CHECK("alloc device mvbits",cudaMalloc(&gd_mvbits, mvbits_size));
	//CUDA_CHECK("alloc device refbits",cudaMalloc(&gd_refbits,refbits_size));
	CUDA_CHECK("alloc device byte_abs",cudaMalloc(&gd_byte_abs, byte_abs_size));

	//获得表头与offset
	int* mvbits_start = mvbits - max_mvd;
	int mvbits_offset = max_mvd;

	unsigned int* byte_abs_start = byte_abs - byte_abs_range/2;
	int byte_abs_offset = byte_abs_range/2;


	//copy array host to device
	CUDA_CHECK("copy spiral_search_x",
		   	cudaMemcpy(gd_spiral_search_x,spiral_search_x,spiral_search_size,cudaMemcpyHostToDevice)
			);
	CUDA_CHECK("copy spiral_search_y",
			cudaMemcpy(gd_spiral_search_y,spiral_search_y,spiral_search_size,cudaMemcpyHostToDevice)
			);
	CUDA_CHECK("copy mvbits" ,cudaMemcpy(gd_mvbits, mvbits_start, mvbits_size, cudaMemcpyHostToDevice));
	//CUDA_CHECK("copy refbits", cudaMemcpy(gd_refbits, refbits, cudaMemcpyHostToDevice));
	CUDA_CHECK("copy byte_abs", cudaMemcpy(gd_byte_abs, byte_abs_start, byte_abs_size, cudaMemcpyHostToDevice));
	//copy offsets
	CUDA_CHECK("copy mvbits offset",
			cudaMemcpyToSymbol(&gd_mvbits_offset,&mvbits_offset,sizeof(mvbits_offset),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy byte_abs_offset",
			cudaMemcpyToSymbol(&gd_byte_abs_offset,&byte_abs_offset,sizeof(mvbits_offset),0,cudaMemcpyHostToDevice));

	//copy other symbol
	int img_width = img->width;
	int img_height = img->height;
	CUDA_CHECK("copy img width",
			cudaMemcpyToSymbol(&gd_picture_width_in_pel, &img_width,sizeof(img_width),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy img height",
			cudaMemcpyToSymbol(&gd_picture_height_in_pel,&img_height,sizeof(img_height),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy max search pos nr",
			cudaMemcpyToSymbol(&gd_max_search_pos_nr,&max_search_points,sizeof(max_search_points),0,cudaMemcpyHostToDevice));
	int width_in_mb = img_width/16;
	int height_in_mb = img_height/16;
	CUDA_CHECK("copy width in mb",
			cudaMemcpyToSymbol(&gd_picture_width_in_mb,&width_in_mb,sizeof(width_in_mb), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK("copy height in mb",
			cudaMemcpyToSymbol(&gd_picture_height_in_mb,&height_in_mb,sizeof(height_in_mb),0,cudaMemcpyHostToDevice));
	int max_mb_nr = width_in_mb * height_in_mb;
	CUDA_CHECK("copy max mb number",
			cudaMemcpyToSymbol(&gd_max_macroblock_nr, &max_mb_nr, sizeof(max_mb_nr), 0, cudaMemcpyHostToDevice));
	int xthread_required = max_mb_nr * max_search_points;
	CUDA_CHECK("copy max thread x required",
			cudaMemcpyToSymbol(&gd_max_threadx_nr_requried, &xthread_required, sizeof(xthread_required), 0,cudaMemcpyHostToDevice)
			);


	//bind texture
	CUDA_CHECK("bind spiral_search_x_texture",
			cudaBindTexture(NULL,
				&g_tex_ref_spiral_search_x,gd_spiral_search_x,
				&g_tex_ref_spiral_search_x.channelDesc,spiral_search_size));
	CUDA_CHECK("bind spiral_search_y_txture",
			cudaBindTexture(NULL,
				&g_tex_ref_spiral_search_y,gd_spiral_search_y,
				&g_tex_ref_spiral_search_y.channelDesc,spiral_search_size));
	CUDA_CHECK("bind mvbits texture",
			cudaBindTexture(NULL, &g_tex_ref_mvbits,gd_mvbits, &g_tex_ref_mvbits.channelDesc, mvbits_size));
	//CUDA_CHECK("bind refbits texture",
	//		cudaBindTexture(NULL, &g_tex_ref_refbits, &g_tex_ref_refbits.channelDesc, refbits_size));
	CUDA_CHECK("bind byte abs texture",
			cudaBindTexture(NULL, &g_tex_ref_byte_abs, gd_byte_abs, &g_tex_ref_byte_abs.channelDesc, byte_abs_size));

	cuda_init_blockSAD(img->max_num_references, search_range, img_width, img_height);
}

//#define CUDA_RefBits(x) tex1Dfetch(g_tex_ref_refbits, x)


extern "C" void cuda_free()
{
	CUDA_CHECK("unbind texture for byte abs ", cudaUnbindTexture(&g_tex_ref_byte_abs));
	CUDA_CHECK("unbind texture for mvbits", cudaUnbindTexture(&g_tex_ref_mvbits));
	CUDA_CHECK("unbind texture for refbits", cudaUnbindTexture(&g_tex_ref_refbits));
	CUDA_CHECK("free device byte abs", cudaFree(&gd_byte_abs));
	CUDA_CHECK("free device mvbits", cudaFree(&gd_mvbits));
	CUDA_CHECK("free device refbits", cudaFree(&gd_refbits));
}



//called after init_img() img->max_num_references is used


extern "C" void cuda_free_blockSAD(int max_num_references)
{
	for(int i = 0; i < max_num_references; ++i)
	{
	}
}


#define CUDA_Byte_ABS(x) tex1Dfetch(g_tex_ref_byte_abs, gd_byte_abs_offset + (x))
#define CUDA_MVBits(x) tex1Dfetch(g_tex_ref_mvbits, gd_mvbits_offset+(x))
#define CUDA_Spiral_Search_X(pos) tex1Dfetch(g_tex_ref_spiral_search_x, pos)
#define CUDA_Spiral_Search_Y(pos) tex1Dfetch(g_tex_ref_spiral_search_y, pos)


//获得所有宏块的所有4x4块的sad
//blockDim(32,16)
__global__ void cuda_setup4x4_block_sad(cudaPitchedPtr block_sad_ptr)
{
	int xid = threadIdx.x + blockIdx.x * blockDim.x;
	int b4x4idx = threadIdx.y;
	if(xid < gd_max_threadx_nr_requried)
	{
		int mb_idx = xid / gd_max_search_pos_nr;
		int mvpos_idx = xid % gd_max_search_pos_nr;
		int b4x_offset = b4x4idx % 4;
		int b4y_offset = b4x4idx / 4;
		//获得搜索中心整数像素精度
		int search_center_x = tex1Dfetch(g_tex_ref_mb_search_center_x, mb_idx) + b4x_offset;
		int search_center_y = tex1Dfetch(g_tex_ref_mb_search_center_y, mb_idx) + b4y_offset;
		int sad = 0;
		int src_x = mb_idx % gd_picture_width_in_mb << 4 + b4x_offset;
		int src_y = mb_idx / gd_picture_width_in_mb << 4 + b4y_offset;
		for(int y = 0; y < 4; ++y)
			for(int x = 0; x < 4; ++x)
				sad += CUDA_Byte_ABS(tex2D(g_tex_ref_imgY_src, src_x + x, src_y + y) 
						- tex2D(g_tex_ref_imgY_ref, search_center_x+x, search_center_y+y));
		CUDA_3D_Element(int, block_sad_ptr, mvpos_idx, BlockOffset4x4+b4x4idx, mb_idx) = sad;
	}
}

extern "C" int 						//  ==> minimum motion cost after search
cudaFastFullPelBlockMotionSearch (pel_t**   orig_pic,     // <--  not used
                              short     ref,          // <--  reference frame (0... or -1 (backward))
                              int       list,
                              int       pic_pix_x,    // <--  absolute x-coordinate of regarded AxB block
                              int       pic_pix_y,    // <--  absolute y-coordinate of regarded AxB block
                              int       blocktype,    // <--  block type (1-16x16 ... 7-4x4) short     pred_mv_x_in_subpel,    // <--  motion vector predictor (x) in sub-pel units short     pred_mv_y_in_subpel,    // <--  motion vector predictor (y) in sub-pel units short*    p_mv_x,         //  --> motion vector (x) - in pel units
                              short*    p_mv_y,         //  --> motion vector (y) - in pel units
                              int       search_range, // <--  1-d search range in pel units
                              int       min_mcost,    // <--  minimum motion cost (cost for center or huge value)
                              int       lambda_factor)       // <--  lagrangian parameter for determining motion cost
{

	int offset_x;
	int offset_y;
	int cand_x;
	int cand_y;
	int mcost;
	//TODO 使用cuda搜索返回最小的motion cost 与运动矢量x，y
	return 0;
}
















/*
texture<imgpel, cudaTextureType2D>  g_tex_ref_imgY_src;

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

	int pel = tex2D(g_tex_ref_imgY_src, getX, getY);
	hpel_temp[ty][tx<<1] = pel << 10;
	pel = 
		20 * (pel + tex2D(g_tex_ref_imgY_src, getX+1, getY))
		-5 * (tex2D(g_tex_ref_imgY_src, getX-1, getY) + tex2D(g_tex_ref_imgY_src, getX+2, getY))
		+ (tex2D(g_tex_ref_imgY_src, getX-2, getY) + tex2D(g_tex_ref_imgY_src, getX + 3, getY));

	hpel_temp[threadIdx.y][threadIdx.x<<1 + 1] = pel << 5;//1024倍
	__syncthreads();//栅栏同步

	if(ty < 4)
	{
		CUDA_2D_Element(imgpel, d_outY4, xin1_4, yin1_4, outY4_pitch) = (imgpel)(hpel_temp[mappedty][tx<<1] >> 10);//整数点
		CUDA_2D_Element(imgpel, d_outY4, xin1_4 + 2, yin1_4, outY4_pitch) = 
			(imgpel)Clip3(0, 255, (hpel_temp[mappedty][tx<<1+1] + 512) >> 10);//整数行水平1/2插值点

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
			
			CUDA_2D_Element(imgpel, d_outY4, xin1_4 + i << 1, yin1_4 + 2, outY4_pitch) = (imgpel)pel;//垂直1/2差值, 原来的32倍
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
	
	CUDA_2D_Element(imgpel, d_outY4, xin1_4, yin1_4, outY4_pitch) = (imgpel)temp[ty][tx];//整数与1/2像素
	CUDA_2D_Element(imgpel, d_outY4, xin1_4+1, yin1_4, outY4_pitch) = (imgpel)h_qpel;//1/4水平
	CUDA_2D_Element(imgpel, d_outY4, xin1_4, yin1_4+1, outY4_pitch) = (imgpel)v_qpel;//1/4垂直
	CUDA_2D_Element(imgpel, d_outY4, xin1_4+1, yin1_4+1, outY4_pitch) = (imgpel)i_qpel;//1/4 倾斜
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
	g_tex_ref_imgY_src.addressMode[0] = cudaAddressModeClamp;//钳位访问
	g_tex_ref_imgY_src.normalized = false;
	CUDA_CHECK("bind imgY src to g_tex_ref_imgY_src",
			cudaBindTextureToArray(&g_tex_ref_imgY_src, d_imgY_src, & g_tex_ref_imgY_src.channelDesc)
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
}*/

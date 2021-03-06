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
#include "memalloc.h"
#include "image.h"
}


#define ThreadPerBlock 256


#define BlockOffset4x4 0
#define BLOCK_TYPE_SAD_COUNT (16+8+8+4+2+2+1) //16_4x4+8_8x4+8_4x8+4_8x8+2_16x8+2_8x16+1_16x16
#define CUDA_ERROR(err, do_what) cuda_h264_error(err, do_what, __FILE__, __LINE__)
#define CUDA_CHECK(do_what, x) \
	do{\
		cudaError_t err; \
		if((err = (x)) != cudaSuccess) \
			CUDA_ERROR(err,do_what); \
	}while(0) 
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

static int g_addup_index[41][16];//i
static int*** g_mv_mean_of_mb;//if
static int g_max_search_points;//i
static int *g_search_picture_done;//if
static int g_block_need_rounded_for_find_min_sad;//i
static int g_max_macroblock_nr;//i

texture<unsigned int, cudaTextureType1D> g_tex_ref_byte_abs;//绝对值表纹理 iu
texture<int,cudaTextureType1D> g_tex_ref_mvbits;//mvbits表纹理		iu
texture<short, 1> g_tex_ref_spiral_search_x;//i
texture<short, 1> g_tex_ref_spiral_search_y;//i
texture<imgpel, 2> g_tex_ref_imgY_src;//iu 
texture<imgpel, 2> g_tex_ref_imgY_ref;// cu 
//
texture<int, 2> g_tex_ref_addup_sad_index;//i
texture<int, 2> g_tex_ref_mv_mean_of_mb;//cu

__constant__ int gd_byte_abs_offset;//i
__constant__ int gd_mvbits_offset;//i
__constant__ int gd_max_search_pos_nr;//i
__constant__ int gd_max_macroblock_nr;//i
__constant__ int gd_picture_width_in_pel;//i
__constant__ int gd_picture_width_in_mb;//i
__constant__ int gd_picture_height_in_pel;//i
__constant__ int gd_picture_height_in_mb;//i
__constant__ int gd_max_threadx_nr_requried;//i

#define CUDA_Byte_ABS(x) tex1Dfetch(g_tex_ref_byte_abs, gd_byte_abs_offset + (x))
#define CUDA_MVBits(x) tex1Dfetch(g_tex_ref_mvbits, gd_mvbits_offset+(x))
#define CUDA_Spiral_Search_X(pos) tex1Dfetch(g_tex_ref_spiral_search_x, pos)
#define CUDA_Spiral_Search_Y(pos) tex1Dfetch(g_tex_ref_spiral_search_y, pos)
#define CUDA_MV_COST(f,s,cx,cy,px,py)   (WEIGHTED_COST(f,CUDA_MVBits(((cx)<<(s))-px)+CUDA_MVBits(((cy)<<(s))-py)))


//[ref][3D ]
//ref维位于host
//后面三维位于device
//depth-> mb (z)
//width-> pos_of_mv (x)
//each AxB block index->height (y)
static cudaPitchedPtr* g_blockSAD;//	if
static cudaArray_t gd_imgY_org_arr;// if
static cudaArray_t gd_addup_sad_index;//	if
static cudaArray_t gd_mv_mean_of_mb;//if
static unsigned int* gd_byte_abs;//设备绝对值表 	if
static int * gd_mvbits;//设备mvbits表	if
static short* gd_spiral_search_x;//if
static short* gd_spiral_search_y;//if
static int* gd_mvcost;//if
static int* gd_mv_pos;//if
static int* gh_mvcost;//if
static int* gh_mv_pos;//if


static void cuda_h264_error(cudaError_t err, const char* do_what, const char* file, int line) 
{
	fprintf(stderr, "in %s at %d line:when %s .For %s\n", file, line, do_what, cudaGetErrorString(err));
	flush_dpb();
	exit(EXIT_FAILURE);
}

//TODO delete debug code
__global__ void addup_sad_index_test(int * d_out)
{
	for(int bidx = 0; bidx < BLOCK_TYPE_SAD_COUNT; ++bidx)
	{
		for(int i = 0; i < 16; ++i)
		{
			d_out[bidx*16 + i] = tex2D(g_tex_ref_addup_sad_index, i, bidx);
		}
	}
}

//TODO delete debug code
static void addup_sad_index_test_case()
{
	int* d_out;
	cudaMalloc(&d_out,sizeof(int)*16 * BLOCK_TYPE_SAD_COUNT);
	int* h_out = (int*)malloc(sizeof(int)*16*BLOCK_TYPE_SAD_COUNT);
	addup_sad_index_test<<<1,1>>>(d_out);
	cudaMemcpy(h_out,d_out,sizeof(int)*16*BLOCK_TYPE_SAD_COUNT,cudaMemcpyDeviceToHost);
	for(int b = 0; b < BLOCK_TYPE_SAD_COUNT; ++b)
	{
		for(int i = 0; i < 16; ++i)
		{
			dbgt("%3d", h_out[b*16+i]);
		}
		dbgt("\n");
	}

	dbgt("src in mem idx\n");
	for(int b = 0; b < BLOCK_TYPE_SAD_COUNT; ++b)
	{
		for(int i = 0; i < 16; ++i)
		{
			dbgt("%3d", g_addup_index[b][i]);
		}
		dbgt("\n");
	}
}

static void cuda_init_addup_idx()
{
#define zero_idx 16
	//4x4
	for(int i = 0; i < 16; ++i)
	{
		g_addup_index[i][0] = i;
		for(int j = 1; j < 16; ++j)
		{
			g_addup_index[i][j] = zero_idx;
		}
	}
	//8x4
	for(int i = 0; i < 8; ++i)
	{
		g_addup_index[i+16][0] = i*2;
		g_addup_index[i+16][1] = i*2+1;
		for(int j = 2; j < 16; ++j)
			g_addup_index[i+16][j] = zero_idx;
	}

	//4x8
	for(int i = 0; i < 8; ++i)
	{
		g_addup_index[i+24][0] = i/4*8 + i%4;
		g_addup_index[i+24][1] = i/4*8 + i%4 + 4;
		for(int j = 2; j < 16; ++j)
			g_addup_index[i+24][j] = zero_idx;
	}
	//8x8	
	for(int i = 0; i < 4; ++i)
	{
		//8x8
		for(int j = 0; j < 4; ++j)
			g_addup_index[i+32][j] = i%2*2 + i/2*8 + j/2*4+j%2;
		for(int j = 4; j < 16; ++j)
			g_addup_index[i+32][j] = zero_idx;
	}

	//16x8
	for(int i = 0; i < 2; ++i)
	{
		for(int j = 0; j < 8; ++j)
			g_addup_index[i+36][j] = i * 8  + j / 4 * 4 + j % 4;
		for(int j = 8; j < 16; ++j)
			g_addup_index[i+36][j] = zero_idx;
	}

	//8x16
	for(int i = 0; i < 2; ++i)
	{
		for(int j = 0; j < 8; ++j)
			g_addup_index[i+38][j] = i * 2 + j / 2 * 4 + j % 2;
		for(int j = 8; j < 16; ++j)
			g_addup_index[i+38][j] = zero_idx;
	}

	//16x16
	for(int i =0 ; i < 16; ++i)
		g_addup_index[40][i] = i;

	cudaChannelFormatDesc arr_desc = cudaCreateChannelDesc<int>();
	CUDA_CHECK("alloc addup array", cudaMallocArray(&gd_addup_sad_index,&arr_desc,16,BLOCK_TYPE_SAD_COUNT,cudaArrayTextureGather));
	CUDA_CHECK("copy addup array to device",
			cudaMemcpy2DToArray(gd_addup_sad_index,0,0,&g_addup_index[0][0],16*sizeof(int),16*sizeof(int),
				BLOCK_TYPE_SAD_COUNT,cudaMemcpyHostToDevice));
	g_tex_ref_addup_sad_index.normalized = false;
	CUDA_CHECK("bind texture of addup table",
			cudaBindTextureToArray(&g_tex_ref_addup_sad_index,gd_addup_sad_index,&g_tex_ref_addup_sad_index.channelDesc)
			);

	//addup_sad_index_test_case();
#undef zero_idx
}

static void cuda_init_blockSAD(int max_num_references, int max_search_points, int img_width, int img_height)
{
	int mb_count = (img_width/16) * (img_height / 16);
	int width_in_bytes = max_search_points*sizeof(short);
	struct cudaExtent block3DExtent = make_cudaExtent(width_in_bytes,BLOCK_TYPE_SAD_COUNT,mb_count);

	if((g_blockSAD = (cudaPitchedPtr*)calloc(sizeof(cudaPitchedPtr), max_num_references)) == NULL)
		no_mem_exit("alloc cudaPitchPtr in host");

	for(int i = 0; i < max_num_references; ++i)
	{
		CUDA_CHECK("alloc 3d blockSAD in device",cudaMalloc3D(&g_blockSAD[i],block3DExtent));
	}
}

//TODO delete debug code
static void cuda_dump_imgY(imgpel* imgY,int width, int height, const char* dumpfile)
{
	dbg_begin(dumpfile)
	{
		for(int y = 0; y < height; ++y)
		{
			for(int x = 0; x < width; ++x)
			{
				dbg("%d\t", imgY[y*width+x]);
			}
			dbg("\n");
		}
	}
	dbg_end();

}

//TODO delete debug code
__global__ void tex_ref_imgY_src_test(imgpel* d_out)
{
	for(int x = 0; x < gd_picture_width_in_pel; ++ x)
		for(int y = 0; y < gd_picture_height_in_pel; ++y) {
			d_out[y * gd_picture_width_in_pel + x] = tex2D(g_tex_ref_imgY_src, x, y);
		}
}

//TODO delete debug code
__global__ void tex_ref_ref_imgY_test(imgpel* d_out)
{	
	for(int x = 0; x < gd_picture_width_in_pel; ++ x)
		for(int y = 0; y < gd_picture_height_in_pel; ++y)
		{
			d_out[y * gd_picture_width_in_pel + x] = tex2D(g_tex_ref_imgY_ref, x, y);
		}

}

//TODO delete debug code
__global__ void tex_ref_mv_mean_of_mb(int* d_out)
{
	for(int i = 0; i < gd_max_macroblock_nr; ++i)
	{
		d_out[i*2 + 0] = tex2D(g_tex_ref_mv_mean_of_mb, 0, i);
		d_out[i*2 + 1] = tex2D(g_tex_ref_mv_mean_of_mb, 1, i);
	}
}

//TODO delete debug code
static void cuda_dump_mv_mean(int* mv, const char* filename)
{
	dbg_begin(filename)
	{
		for(int mb = 0; mb < g_max_macroblock_nr; ++mb)
		{
			dbg("mb idx = %d mean mv (%d, %d)\n", mb, mv[mb*2], mv[mb*2+1]);
		}
	}
	dbg_end();

}

//TODO delete debug code
static void cuda_mv_mean_of_mv_texture_test(const char* filename)
{
	int* d_out;
	CUDA_CHECK("alloc mv mean out",
			cudaMalloc(&d_out,sizeof(int)*g_max_macroblock_nr*2)
			);
	int* h_out = (int*)malloc(sizeof(int)*2*g_max_macroblock_nr);
	tex_ref_mv_mean_of_mb<<<1,1>>>(d_out);
	CUDA_CHECK("copy d_out", cudaMemcpy(h_out, d_out,sizeof(int)*2*g_max_macroblock_nr,cudaMemcpyDeviceToHost));
	cuda_dump_mv_mean(h_out, filename);
	CUDA_CHECK("free d_out", cudaFree(d_out));
	free(h_out);
}

//TODO delete debug code
static void cuda_dump_src_imgY_for_texture(const char* filename)
{
	imgpel* d_imgY;
	CUDA_CHECK("alloc d_imgY", cudaMalloc(&d_imgY, sizeof(imgpel)*img->width*img->height));
	tex_ref_imgY_src_test<<<1,1>>>(d_imgY);
	imgpel* h_imgY = (imgpel*)malloc(sizeof(imgpel)*img->width*img->height);
	CUDA_CHECK("copy test imgY", cudaMemcpy(h_imgY,d_imgY,sizeof(imgpel)*img->width*img->height,cudaMemcpyDeviceToHost));
	cuda_dump_imgY(h_imgY, img->width, img->height, filename);
	CUDA_CHECK("free d_imgY", cudaFree(d_imgY));
	free(h_imgY);
}

//TODO delete debug code
static void cuda_dump_ref_imgY_for_texture(const char* filename)
{
	imgpel* d_imgY;
	CUDA_CHECK("alloc d_imgY", cudaMalloc(&d_imgY, sizeof(imgpel)*img->width*img->height));
	tex_ref_ref_imgY_test<<<1,1>>>(d_imgY);
	imgpel* h_imgY = (imgpel*)malloc(sizeof(imgpel)*img->width*img->height);
	CUDA_CHECK("copy test imgY", cudaMemcpy(h_imgY,d_imgY,sizeof(imgpel)*img->width*img->height,cudaMemcpyDeviceToHost));
	cuda_dump_imgY(h_imgY, img->width, img->height, filename);
	CUDA_CHECK("free d_imgY", cudaFree(d_imgY));
	free(h_imgY);
}

static void get_mean_mvs(int max_mb_nr, int mb_in_width, int** mvs, StorablePicture* refpic)
{
	for(int i = 0; i < max_mb_nr; ++i)
	{
		int meanx = 0; 
		int meany = 0;
		int xbase = i % mb_in_width * 4;
		int ybase = i / mb_in_width * 4;
		for(int j = 0; j < 16; ++j)
		{
			int b4x = xbase + j%4;
			int b4y = ybase + j/4;
			meanx += refpic->mv[0][b4y][b4x][0];
			meany += refpic->mv[0][b4y][b4x][1];
		}
		mvs[i][0] = meanx/16;
		mvs[i][1] = meany/16;
	}
}

extern "C" void cuda_copy_one_frame_and_bind_texture(imgpel* imgY,int width, int height)
{
	CUDA_CHECK("copy imgY", 
			cudaMemcpy2DToArray(gd_imgY_org_arr,0,0,imgY,width*sizeof(imgpel),width*sizeof(imgpel),height,cudaMemcpyHostToDevice));
	g_tex_ref_imgY_src.addressMode[0] = g_tex_ref_imgY_src.addressMode[1] = cudaAddressModeClamp;
	g_tex_ref_imgY_src.normalized = false;
	CUDA_CHECK("bind imgY texture",
			cudaBindTextureToArray(&g_tex_ref_imgY_src,gd_imgY_org_arr,&g_tex_ref_imgY_src.channelDesc)
			);
}

//初始化cuda motion search
//call after Init_Motion_Search_Module ()
extern "C" void cuda_init_motion_search_module()
{
	int search_range               = input->search_range;
	g_max_search_points            = max(9, (2*search_range+1)*(2*search_range+1));
	int number_of_subpel_positions = 4 * (2*search_range+3);
	int max_mv_bits                = 3 + 2 * (int)ceil (log(number_of_subpel_positions+1) / log(2) + 1e-10);
	int max_mvd                    = (1<<( max_mv_bits >>1))-1;

	size_t spiral_search_size = g_max_search_points*sizeof(short);
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
			cudaMemcpyToSymbol(gd_mvbits_offset,&mvbits_offset,sizeof(mvbits_offset),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy byte_abs_offset",
			cudaMemcpyToSymbol(gd_byte_abs_offset,&byte_abs_offset,sizeof(mvbits_offset),0,cudaMemcpyHostToDevice));

	//copy other symbol
	int img_width = img->width;
	int img_height = img->height;
	CUDA_CHECK("copy img width",
			cudaMemcpyToSymbol(gd_picture_width_in_pel, &img_width,sizeof(img_width),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy img height",
			cudaMemcpyToSymbol(gd_picture_height_in_pel,&img_height,sizeof(img_height),0,cudaMemcpyHostToDevice));
	CUDA_CHECK("copy max search pos nr",
			cudaMemcpyToSymbol(gd_max_search_pos_nr,&g_max_search_points,sizeof(g_max_search_points),0,cudaMemcpyHostToDevice));
	int width_in_mb = img_width/16;
	int height_in_mb = img_height/16;
	CUDA_CHECK("copy width in mb",
			cudaMemcpyToSymbol(gd_picture_width_in_mb,&width_in_mb,sizeof(width_in_mb), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK("copy height in mb",
			cudaMemcpyToSymbol(gd_picture_height_in_mb,&height_in_mb,sizeof(height_in_mb),0,cudaMemcpyHostToDevice));
	int max_mb_nr = width_in_mb * height_in_mb;
	g_max_macroblock_nr = max_mb_nr;
	CUDA_CHECK("copy max mb number",
			cudaMemcpyToSymbol(gd_max_macroblock_nr, &max_mb_nr, sizeof(max_mb_nr), 0, cudaMemcpyHostToDevice));
	int xthread_required = max_mb_nr * g_max_search_points;
	CUDA_CHECK("copy max thread x required",
			cudaMemcpyToSymbol(gd_max_threadx_nr_requried, &xthread_required, sizeof(xthread_required), 0,cudaMemcpyHostToDevice)
			);

	g_tex_ref_spiral_search_x.normalized = false;
	//bind texture
	CUDA_CHECK("bind spiral_search_x_texture",
			cudaBindTexture(NULL,
				&g_tex_ref_spiral_search_x,gd_spiral_search_x,
				&g_tex_ref_spiral_search_x.channelDesc,spiral_search_size));
	g_tex_ref_spiral_search_y.normalized = false;
	CUDA_CHECK("bind spiral_search_y_txture",
			cudaBindTexture(NULL,
				&g_tex_ref_spiral_search_y,gd_spiral_search_y,
				&g_tex_ref_spiral_search_y.channelDesc,spiral_search_size));
	g_tex_ref_mvbits.normalized = false;
	CUDA_CHECK("bind mvbits texture",
			cudaBindTexture(NULL, &g_tex_ref_mvbits,gd_mvbits, &g_tex_ref_mvbits.channelDesc, mvbits_size));
	//CUDA_CHECK("bind refbits texture",
	//		cudaBindTexture(NULL, &g_tex_ref_refbits, &g_tex_ref_refbits.channelDesc, refbits_size));
	g_tex_ref_byte_abs.normalized = false;
	CUDA_CHECK("bind byte abs texture",
			cudaBindTexture(NULL, &g_tex_ref_byte_abs, gd_byte_abs, &g_tex_ref_byte_abs.channelDesc, byte_abs_size));

	cuda_init_blockSAD(img->max_num_references, g_max_search_points, img_width, img_height);
	
	cuda_init_addup_idx();

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<imgpel>();
	CUDA_CHECK("alloc imgY cuda array",
			cudaMallocArray(&gd_imgY_org_arr,&desc,img->width,img->height,cudaArrayTextureGather));

	g_block_need_rounded_for_find_min_sad = (g_max_search_points + ThreadPerBlock - 1)/ThreadPerBlock;
	CUDA_CHECK("alloc mvcost on device",
			cudaMalloc(&gd_mvcost,sizeof(int)*g_block_need_rounded_for_find_min_sad));
	CUDA_CHECK("alloc mv pos on device",
			cudaMalloc(&gd_mv_pos, sizeof(int)*g_block_need_rounded_for_find_min_sad));

	gh_mvcost = (int*)malloc(sizeof(int)*g_block_need_rounded_for_find_min_sad);
	gh_mv_pos = (int*)malloc(sizeof(int)*g_block_need_rounded_for_find_min_sad);
	CUDA_CHECK("alloc device mvcost",cudaMalloc(&gd_mvcost,sizeof(int)*g_block_need_rounded_for_find_min_sad));
	CUDA_CHECK("alloc device mv_pos", cudaMalloc(&gd_mv_pos, sizeof(int)*g_block_need_rounded_for_find_min_sad));

	get_mem3Dint(&g_mv_mean_of_mb,img->max_num_references, g_max_macroblock_nr,2);

	cudaChannelFormatDesc mv_mean_desc = cudaCreateChannelDesc<int>();
	CUDA_CHECK("alloc device array mv mean of mb",
			cudaMallocArray(&gd_mv_mean_of_mb,&mv_mean_desc,2,g_max_macroblock_nr,cudaArrayTextureGather)
			);
	g_search_picture_done = (int*)malloc(sizeof(int)*img->max_num_references);
}

extern "C" void cuda_free()
{
	CUDA_CHECK("unbind texture for byte abs ", cudaUnbindTexture(&g_tex_ref_byte_abs));
	CUDA_CHECK("unbind texture for mvbits", cudaUnbindTexture(&g_tex_ref_mvbits));
	CUDA_CHECK("unbind texture for imgY", cudaUnbindTexture(&g_tex_ref_imgY_src));
	CUDA_CHECK("unbind texture for spiral_search_x", cudaUnbindTexture(&g_tex_ref_spiral_search_x));
	CUDA_CHECK("unbind texture for spiral_search_y", cudaUnbindTexture(&g_tex_ref_spiral_search_y));
	CUDA_CHECK("unbind texture for imgY_ref", cudaUnbindTexture(&g_tex_ref_imgY_ref));
	CUDA_CHECK("unbind texture for addup sad index", cudaUnbindTexture(&g_tex_ref_addup_sad_index));



	CUDA_CHECK("free device byte abs", cudaFree(gd_byte_abs));
	CUDA_CHECK("free device mvbits", cudaFree(gd_mvbits));
	CUDA_CHECK("free device spiral_search_x", cudaFree(gd_spiral_search_x));
	CUDA_CHECK("free device spiral_search_y", cudaFree(gd_spiral_search_y));
	CUDA_CHECK("free device mvcost", cudaFree(gd_mvcost));
	CUDA_CHECK("free device mv_pos", cudaFree(gd_mv_pos));
	free(gh_mvcost);
	free(gh_mv_pos);

	for(int i = 0; i < img->max_num_references; ++i)
	{
		CUDA_CHECK("free block sad 3D array",cudaFree(g_blockSAD[i].ptr));
	}
	free(g_blockSAD);

	CUDA_CHECK("free device array imgY org", cudaFreeArray(gd_imgY_org_arr));
	CUDA_CHECK("free device array addup sad idx", cudaFreeArray(gd_addup_sad_index));
	CUDA_CHECK("free device array mean of mb", cudaFreeArray(gd_mv_mean_of_mb));


	free_mem3Dint(g_mv_mean_of_mb,img->max_num_references);
	free(g_search_picture_done);
}

//add before encode_one_frame
extern "C" void cuda_begin_encode_frame()
{
	for(int i = 0; i < img->max_num_references; ++i)
		g_search_picture_done[i] = 0;
}
extern "C" void cuda_free_device_imgY(cudaArray_t arr)
{
	CUDA_CHECK("free storeable picture imgY", cudaFreeArray(arr));
}
extern "C" void cuda_alloc_device_imgY(cudaArray_t* arr)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<imgpel>();
	CUDA_CHECK("alloc storable picture imgY",
			cudaMallocArray(arr,&desc,img->width,img->height,cudaArrayTextureGather)
			);
}
extern "C" void cuda_end_encode_frame()
{
	CUDA_CHECK("copy imgY to device array after encode frame",
			cudaMemcpy2DToArray(
				enc_picture->d_imgY,0,0,
				enc_picture->imgY[0],
				img->width*sizeof(imgpel),
				img->width*sizeof(imgpel),
				img->height,
				cudaMemcpyHostToDevice)
			);
}

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
		int b4x_offset = b4x4idx % 4 * 4;
		int b4y_offset = b4x4idx / 4 * 4;
		//获得搜索中心整数像素精度
		int src_x = mb_idx % gd_picture_width_in_mb * 16 + b4x_offset;
		int src_y = mb_idx / gd_picture_width_in_mb * 16 + b4y_offset;
		int search_center_x = tex2D(g_tex_ref_mv_mean_of_mb, 0, mb_idx) + src_x + CUDA_Spiral_Search_X(mvpos_idx);
		int search_center_y = tex2D(g_tex_ref_mv_mean_of_mb, 1, mb_idx) + src_y + CUDA_Spiral_Search_Y(mvpos_idx);
		unsigned short sad = 0;
		for(int y = 0; y < 4; ++y)
		{
			for(int x = 0; x < 4; ++x)
			{
				int diff = tex2D(g_tex_ref_imgY_src, src_x + x, src_y + y) - 
					tex2D(g_tex_ref_imgY_ref,search_center_x+x, search_center_y+y);
				sad += CUDA_Byte_ABS(diff);
			}

		}
		CUDA_3D_Element(unsigned short, block_sad_ptr, mvpos_idx, BlockOffset4x4+b4x4idx, mb_idx) = sad;
	}
}

//一个Block对应一个MB 41 thread
//gird (mb_max*search_range)
__global__ void cuda_addup_large_block_sads(cudaPitchedPtr block_sad_ptr)
{
	int bidx = threadIdx.x;
	int pos_start = gridDim.y * blockIdx.y;
	int pos_end = pos_start + gridDim.y;
	int mb_idx = blockIdx.x;
	int sad;
	__shared__ int sad_cache[17];
	if(bidx == 0) sad_cache[16] = 0;
	for(int pos = pos_start; pos < pos_end; ++pos)
	{
		sad = 0;
		if(bidx < 16)
			sad_cache[bidx] = CUDA_3D_Element(unsigned short, block_sad_ptr, pos, bidx, mb_idx);
		__syncthreads();
		for(int i = 0; i < 16; ++i)
		{
			sad += sad_cache[tex2D(g_tex_ref_addup_sad_index, i, bidx)];
		}
		__syncthreads();
		CUDA_3D_Element(unsigned short, block_sad_ptr, pos, bidx, mb_idx) = sad;
	}
}

//TODO delete debug code
static void dbg_block_sad_of_mb(const char* filename, int ref, int mb)
{
	dbg_begin(filename)
	{
		static void *h_block_sad = NULL;
		int pitch = g_blockSAD[0].pitch;
		int ysize = g_blockSAD[0].ysize;
		size_t slice_pitch = pitch*ysize;
		if(h_block_sad == NULL)
		{
			h_block_sad = malloc(slice_pitch * g_max_macroblock_nr);
		}
		CUDA_CHECK("copy test block sad\n", 
				cudaMemcpy(h_block_sad,g_blockSAD[ref].ptr,slice_pitch*g_max_macroblock_nr,cudaMemcpyDeviceToHost));

#define Ele3D(type, base_addr, x, y, z, pitch, ysize) ((type*)((char*)(base_addr) + (pitch)*(ysize)*(z) + (pitch)*(y)))[x]
		for(int pos = 0; pos < g_max_search_points; ++pos)
		{
			for(int j = 0; j < BLOCK_TYPE_SAD_COUNT; ++j)
				dbg("%d\t", Ele3D(unsigned short, h_block_sad, pos, j, mb, pitch, ysize));
			dbg("\n");
		}
		dbg("\n");
	}
	dbg_end();
#undef Ele3D

}

//TODO delete debug code
static void dbg_block_sad(const char* filename, int ref, int mb_loop_max, int pos_loop_max, int blc_idx_loop_max)
{
	dbg_begin(filename)
	{
		static void *h_block_sad = NULL;
		int pitch = g_blockSAD[0].pitch;
		int ysize = g_blockSAD[0].ysize;
		size_t slice_pitch = pitch*ysize;
		if(h_block_sad == NULL)
		{
			h_block_sad = malloc(slice_pitch * g_max_macroblock_nr);
		}
		CUDA_CHECK("copy test block sad\n", 
				cudaMemcpy(h_block_sad,g_blockSAD[ref].ptr,slice_pitch*g_max_macroblock_nr,cudaMemcpyDeviceToHost));

#define Ele3D(type, base_addr, x, y, z, pitch, ysize) ((type*)((char*)(base_addr) + (pitch)*(ysize)*(z) + (pitch)*(y)))[x]
		for(int i = 0; i < mb_loop_max; ++i)
		{
			dbg("current mb idx = %d\n", i);
			for(int pos = 0; pos < pos_loop_max; ++pos)
			{
				for(int j = 0; j < blc_idx_loop_max; ++j)
					dbg("%d\t", Ele3D(unsigned short, h_block_sad, pos, j, i, pitch, ysize));
				dbg("\n");
			}
			dbg("\n");
		}
	}
	dbg_end();
#undef Ele3D

}

static void cuda_addup_large_block(int list,int ref)
{
	int sqr_max_points = (int)(sqrt((double)g_max_search_points) + 0.5);
	dim3 grid_dim(g_max_macroblock_nr, sqr_max_points);
	dim3 block_dim(41);
	cuda_addup_large_block_sads<<<grid_dim, block_dim>>>(g_blockSAD[ref]);
	//dbg_block_sad("block_addup_large.txt", ref, g_max_macroblock_nr, g_max_search_points, 41);
}
//thread -> pos
__global__ void cuda_find_min_mvcost(
		cudaPitchedPtr block_sad_ptr, 
		int lambda_factor,
		int mb_idx,
		int block_AxB_idx,
		int cand_x,
		int cand_y,
		int pred_mv_x,
		int pred_mv_y,
		int* d_mcost,
		int* d_mv_pos
		)
{
	int tx = threadIdx.x;
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	int op1,op2, idx;
	__shared__ int mcost[ThreadPerBlock];
	__shared__ int midx[ThreadPerBlock];
	//for loop to find min cost & idx
	cand_x += CUDA_Spiral_Search_X(pos);
	cand_y += CUDA_Spiral_Search_Y(pos);
	mcost[tx] = INT_MAX;//初始化
		//CUDA_MV_COST(lambda_factor, 2, cand_x, cand_y, pred_mv_x, pred_mv_y) + 
	//	CUDA_3D_Element(unsigned short, block_sad_ptr, pos, block_AxB_idx, mb_idx);
	midx[tx] = pos;
	if(pos < gd_max_search_pos_nr)
	{
		mcost[tx] = CUDA_3D_Element(unsigned short, block_sad_ptr, pos, block_AxB_idx, mb_idx) +
			CUDA_MV_COST(lambda_factor, 2, cand_x, cand_y, pred_mv_x, pred_mv_y);
	}
	__syncthreads();
	int threads = ThreadPerBlock/2;
	while(threads)
	{
		if(tx < threads)
		{
			op1 = mcost[tx];
			op2 = mcost[tx + threads];
			idx = midx[tx + threads];
		}
		__syncthreads();//读取同步
		if(tx < threads && op1 > op2)
		{
			mcost[tx] = op2;
			midx[tx] = idx;
		}
		__syncthreads();//写同步
		threads /= 2;
	}
	if(tx == 0)
	{
		d_mcost[blockIdx.x] = mcost[0];
		d_mv_pos[blockIdx.x] = midx[0];
	}
}

//TODO 添加参数验证代码
extern "C" void cuda_validate_arguments()
{
	//TODO add code to validate arguments
}



//pass
static void cuda_setup_block4x4(int list,int ref)
{
	g_tex_ref_imgY_ref.addressMode[0] = g_tex_ref_imgY_ref.addressMode[1] = cudaAddressModeClamp;
	g_tex_ref_imgY_ref.normalized = false;
	CUDA_CHECK(
			"bind refrences imgY to texture",
			cudaBindTextureToArray(&g_tex_ref_imgY_ref,listX[list][ref]->d_imgY,&g_tex_ref_imgY_ref.channelDesc)
			);//绑定reference 纹理

	//获取mb 的mean mv 并拷贝到数组绑定纹理
	get_mean_mvs(g_max_macroblock_nr, img->width/16, g_mv_mean_of_mb[ref], listX[list][ref]);
	CUDA_CHECK("copy mv mean array",
			cudaMemcpy2DToArray(gd_mv_mean_of_mb, 0, 0,
				g_mv_mean_of_mb[ref][0],sizeof(int)*2,sizeof(int)*2,
				g_max_macroblock_nr,cudaMemcpyHostToDevice)
			);
	g_tex_ref_mv_mean_of_mb.normalized = false;
	CUDA_CHECK("bind texture of mv means",
			cudaBindTextureToArray(&g_tex_ref_mv_mean_of_mb,gd_mv_mean_of_mb,&g_tex_ref_mv_mean_of_mb.channelDesc)
			);

	dim3 dim_block(ThreadPerBlock/16,16);
	int block4x4_total = (img->width/4) * (img->height/4);
	int thread_need_total = block4x4_total * g_max_search_points;
	int block_need_rounded = (thread_need_total + ThreadPerBlock - 1)/ThreadPerBlock;

	//kernel之间是串行执行的
	cuda_setup4x4_block_sad<<<block_need_rounded,dim_block>>>(g_blockSAD[ref]);
	//dbg_block_sad("block_sad_gpu.txt", ref, g_max_macroblock_nr, g_max_search_points, 16);
}

//TODO delete debug code
static void cuda_setup_block4x4_and_addup_cpu(int list, int ref)
{
	StorablePicture* refpic = listX[list][ref];
	static int*** blockSad = NULL;
	if(blockSad == NULL)
	{
		get_mem3Dint(&blockSad,g_max_macroblock_nr,BLOCK_TYPE_SAD_COUNT,g_max_search_points);
	}
	for(int mb_idx = 0; mb_idx < g_max_macroblock_nr; ++mb_idx)
	{
		int mb_offset_x = mb_idx % (img->width/16) * 16;
		int mb_offset_y = mb_idx / (img->width/16) * 16;
		for(int b4x4_idx = 0; b4x4_idx < 16; ++b4x4_idx)
		{
			int inner_x = b4x4_idx % 4 * 4;
			int inner_y = b4x4_idx / 4 * 4;
			int src_x = mb_offset_x + inner_x;
			int src_y = mb_offset_y + inner_y;
			for(int pos = 0; pos < g_max_search_points; ++pos)
			{
				int ref_x = src_x + spiral_search_x[pos] + g_mv_mean_of_mb[ref][mb_idx][0];
				int ref_y = src_y + spiral_search_y[pos] + g_mv_mean_of_mb[ref][mb_idx][1];
				int sad = 0;
				for(int i = 0; i < 4; ++i)
					for(int j = 0; j < 4; ++j)
					{
						sad += byte_abs[imgY_org[src_y + i][src_x + j] - 
							refpic->imgY[Clip3(0, img->height-1,ref_y + i)][Clip3(0, img->width-1,ref_x+j)]];
					}
				blockSad[mb_idx][BlockOffset4x4+b4x4_idx][pos] = sad;
			}
		}
	}

	dbg_begin("block_sad_cpu.txt")
	{
		for(int i = 0; i < g_max_macroblock_nr; ++i)
		{
			dbg("current mb idx = %d\n", i);
			for(int pos = 0; pos < g_max_search_points; ++pos)
			{
				for(int j = 0; j < 16; ++j)
					dbg("%d\t", blockSad[i][j][pos]);

				dbg("\n");
			}
			dbg("\n");
		}
	}
	dbg_end();
}
extern "C" int                                                   //  ==> minimum motion cost after search
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
                              int       lambda_factor)       // <--  lagrangian parameter for determining motion cost
{
	int best_pos = 0;
	int mb_idx = img->current_mb_nr;
	static int block_type_offset[] =
	{
		-1, //16x16 B frame, not support
		40,	//16x16 
		36, //16x8
		38, //8x16
		24, //8x8
		16, //8x4
		24, //4x8
		0	//4x4
	};
	if(list != 0)
	{
		fprintf(stderr, "sorry! motion search using cuda now only support list 0 ,P picture.\n");
		exit(1);
	}

	if(!g_search_picture_done[ref])
	{
		//整个picture 4x4运动搜索
		//整个picture 合成大块sad
		cuda_setup_block4x4(list, ref);
		cuda_addup_large_block(list,ref);
		//cuda_setup_block4x4_and_addup_cpu(list,ref);
		g_search_picture_done[ref] = 1;
	}

	//获得最小mcost与运动矢量
	int axb_block_x_in_mb = pic_pix_x - mb_idx % (img->width / 16) * 16;
	int axb_block_y_in_mb = pic_pix_y - mb_idx / (img->width / 16) * 16;
	int blk_width = input->blc_size[blocktype][0];
	int blk_height = input->blc_size[blocktype][1];
	int block_inner_offset = axb_block_y_in_mb / blk_height * (16/blk_width) + axb_block_x_in_mb / blk_width;

	cuda_find_min_mvcost<<<g_block_need_rounded_for_find_min_sad,ThreadPerBlock>>>(
			g_blockSAD[ref],
			lambda_factor,
			mb_idx,
			block_type_offset[blocktype] + block_inner_offset,
			g_mv_mean_of_mb[ref][mb_idx][0],
			g_mv_mean_of_mb[ref][mb_idx][1],
			pred_mv_x,
			pred_mv_y,
			gd_mvcost,//out
			gd_mv_pos//out
			);

	CUDA_CHECK("move mvcost from device to host",
			cudaMemcpy(gh_mvcost,gd_mvcost,sizeof(int)*g_block_need_rounded_for_find_min_sad,cudaMemcpyDeviceToHost));
	CUDA_CHECK("move mv pos from device to host",
			cudaMemcpy(gh_mv_pos,gd_mv_pos,sizeof(int)*g_block_need_rounded_for_find_min_sad,cudaMemcpyDeviceToHost));
	/*
	dbg_begin("cand_min_cost")
	{
		for(int i = 0; i < g_block_need_rounded_for_find_min_sad; ++i)
		{
			dbg("%d\t", gh_mvcost[i]);
		}
	}
	dbg_end();

	dbg_begin("cand_min_pos")
	{
		for(int i = 0; i < g_block_need_rounded_for_find_min_sad; ++i)
		{
			dbg("%d\t", gh_mv_pos[i]);
		}
	}
	dbg_end();*/


	for(int i = 0; i < g_block_need_rounded_for_find_min_sad; ++i)
	{
		if(min_mcost > gh_mvcost[i])
		{
			min_mcost = gh_mvcost[i];
			best_pos = gh_mv_pos[i];
		}
	}
	*mv_x = g_mv_mean_of_mb[ref][mb_idx][0] + spiral_search_x[best_pos];
	*mv_y = g_mv_mean_of_mb[ref][mb_idx][1] + spiral_search_y[best_pos];
	return min_mcost;
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

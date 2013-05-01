#include<cuda.h>
#include "defines.h"
#include "cuda_h264.h"

//#define IMG_PAD_SIZE 4

//#define imgpel unsigned short


//#define Clip3(a,b,x) (x>a?(x<b?x:b):a)
//#define min(a,b) (a>b?b:a)
//#define BLOCK_SIZE 8


//thread 4x4 in one MB 
//16x16 in 1/4 pel  -> one Block
__global__ void half_pel_filter_frist_run(int width, int height, int paded4width, int paded4height, imgpel* d_imgY, imgpel* d_outY4)
{
	int storeX = blockIdx.x * 16 + threadIdx.x * 4 + 2;
	int storeY = blockIdx.y * 16 + threadIdx.y * 4 + 2;
	int getX = (storeX - 16)/4;
	int getY = (storeY - 16)/4;
	getX = Clip3(0, width-1, getX);
	getY = Clip3(0, height-1, getY);
	int intpel = d_imgY[getY * width + getX];
	int pel = 
		20 * (intpel + d_imgY[getY * width + Clip3(0, width-1, getX+1)]) -
		5 * (d_imgY[getY * width + Clip3(0,width-1, getX-1)] + d_imgY[getY * width + Clip3(0,width-1, getX+2)]) +
		(d_imgY[getY * width + Clip3(0, width-1, getX-2)] + d_imgY[getY * width + Clip3(0, width-1, getX + 3)]);

	d_outY4[storeY * paded4width + storeX - 2] = intpel;//整数像素点直接赋值
	d_outY4[storeY * paded4width + storeX] = pel;//水平差值，原来的32倍

	pel = 20 * (intpel + d_imgY[Clip3(0, height-1, getY+1)*width + getX]) -
		5 * (d_imgY[Clip3(0, height-1, getY-1) * width + getX] + d_imgY[Clip3(0, height-1, getY+2) * width + getX]) + 
		(d_imgY[Clip3(0, height-1, getY-2) * width + getX] + d_imgY[Clip3(0, height-1, getY+3) * width + getX]);

	d_outY4[(storeY-2)*paded4width + storeX - 2] = pel;//垂直1/2差值, 原来的32倍
}

//thread 4x4 in one MB
//16x16 in 1/4 pel -> one Block
__global__ void half_pel_filter_second_run(int paded4width,int paded4height, imgpel* d_outY4)
{
	int storeX = blockIdx.x * 16 + threadIdx.x + 2;
	int storeY = blockIdx.y * 16 + threadIdx.y + 2;
	int pel = //all value has bing * 32
		20 * (d_outY4[Clip3(0, paded4height-1, storeY-2)*paded4width+storeX] + d_outY4[Clip3(0, paded4height-1, storeY+2)*paded4width+storeX]) -
		5 * (d_outY4[Clip3(0, paded4height-1, storeY-6)*paded4width+storeX] + d_outY4[Clip3(0, paded4height-1, storeY+6)*paded4width+storeX]) +
		(d_outY4[Clip3(0,paded4height-1, storeY-10)*paded4width+storeX] + d_outY4[Clip3(0,paded4height-1, storeY+10)*paded4width+storeX]);
	d_outY4[storeY*paded4width+storeX] = (pel + 16) >> 5;//部分精度损失
}
//thread 8x8 in one MB
//one 16x16 in 1/4 -> one Block
__global__ void quarter_pel_filter(int paded4width, int paded4height, imgpel* d_outY4)
{
	__shared__ int temp[9][9];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x * 16 + tx << 1;
	int y = blockIdx.y * 16 + ty << 1;
	temp[ty][tx] = d_outY4[y*paded4width + x] << ((ty%2 == 0 && tx % 2 == 0) * 5); //整数像素点扩大32倍，与后面统一
	if(tx == 7)
		temp[ty][8] = d_outY4[y*paded4width+ min(paded4width-2,x+2)] << ((ty%2 == 0) * 5);
	if(ty == 7)
		temp[8][tx] = d_outY4[min(paded4height-2,y+2)*paded4width+x] << ((tx%2 == 0) * 5);

	__syncthreads();//栅栏同步

	int h_qpel = temp[ty][tx];//水平1/4像素
	h_qpel = (h_qpel + temp[ty][tx+1] + 1) >> 6;
	int v_qpel = temp[ty][tx];//垂直1/4像素
	v_qpel = (v_qpel + temp[ty+1][tx] + 1) >> 6;

	int i_qpel;//倾斜1/4像素
	int off = (tx%2== 0 && ty%2==0);
	i_qpel = (temp[ty+off][tx] + temp[ty+1-off][tx+1] + 1) >> 6;
	/*
	if(tx % 2 == 0 && ty % 2 == 0)
	{//'/'
		i_qpel = (temp[ty+1][tx] + temp[ty][tx+1] + 1) >> 6;
	}
	else 
	{//'\'
		i_qpel = (temp[ty][tx] + temp[ty+1][tx+1] + 1) >> 6;
	}*/

	d_outY4[y*paded4width+x] = temp[ty][tx] >> 5;//整数与1/2像素
	d_outY4[y*paded4width+x+1] = h_qpel;//1/4水平
	d_outY4[(y+1)*paded4width + x] = v_qpel;//1/4垂直
	d_outY4[(y+1)*paded4width + x + 1] = i_qpel;//1/4 倾斜
}


extern "C" 
void quarter_filter(imgpel** imgY, int width, int height, imgpel** outY4)
{
	imgpel* d_imgY;
	int paded4width = (width + IMG_PAD_SIZE * 2) * 4;
	int paded4height = (height + IMG_PAD_SIZE * 2) * 4;

	size_t imgY_size = width*height*sizeof(imgpel);
	cudaMalloc(&d_imgY, imgY_size);
	cudaMemcpy(d_imgY, imgY[0], imgY_size, cudaMemcpyHostToDevice);

	imgpel* d_outY4;
	size_t out4Size = paded4width*paded4height *sizeof(imgpel);
	cudaMalloc(&d_outY4, out4Size);


	dim3 helf_pel_block_dim(4,4);
	dim3 quarter_pel_block_dim(8,8);
	dim3 dimGrid(paded4width/16, paded4height/16);
	//1/2像素第一轮
	half_pel_filter_frist_run<<<dimGrid, helf_pel_block_dim>>>
		(width, height, paded4width, paded4height, d_imgY, d_outY4);
	//1/2像素第二轮
	half_pel_filter_second_run<<<dimGrid, helf_pel_block_dim>>>
		(paded4width, paded4height, d_outY4);
	//1/4像素
	quarter_pel_filter<<<dimGrid, quarter_pel_block_dim>>>
		(paded4width, paded4height, d_outY4);

	cudaMemcpy(outY4[0], d_outY4, out4Size, cudaMemcpyDeviceToHost);
	cudaFree(d_imgY);
	cudaFree(d_outY4);
}


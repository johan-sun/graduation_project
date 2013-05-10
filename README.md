graduation project
==================

my graduation project of improve JM encoder by using cuda

### 算法代码

* cuda\_h264.cu
* cuda\_h264.h

### 修改原始文件

* mbuffer.h
	* storable_picture struct line 53, add cudaArray_t for device imgY
* mbuffer.c 
	* alloc\_storable\_picture function line 404-405, alloc device array for imgY.
   	* free\_storable\_picture function line 532-537, free device array for imgY
* mv-search.c.
	* BlockMotionSearch function line 2881-2887, add cuda FS code for getting min_mcost.
* image.c
	* encode_one_frame function line 335, add function to copy imgY org to device array & bind it to the texture.
* lencod.c
	* main function line 149, add function to validate for cuda
	* main function line 226, add function to init cuda motion search
	* main function line 383, add callback before encode one frame
	* main function line 385, add callback after encode one frame


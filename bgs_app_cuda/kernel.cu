#include <cuda_runtime.h> 
#include "vibe.h"
#include <opencv2/core/core.hpp>
#include <random>

extern int host_img_size[2];
extern int host_samples;
extern int host_channels;
extern int host_pixel_neighbor;
extern int host_distance_threshold;
extern int host_matching_threshold;
extern int host_update_factor;

extern unsigned int host_rng[RANDOM_BUFFER_SIZE];
extern int host_rng_idx;


int rnd_pos[2];
unsigned int* dev_rng;
//extern "C" cv::Vec2i getRndNeighbor(int i, int j);

#define thread_num_per_block 256
#define block_num 32 //32 is optimal block_num for this problem

inline void checkCudaErrors(cudaError err)//错误处理函数
{
	if (cudaSuccess != err)
	{
	    fprintf(stderr, "CUDA Runtime API error: %s.\n", cudaGetErrorString(err));
	    return;
	}
}

__device__ int dev_max(int i, int j)
{
	if(i>j)
		return i;
	else
		return j;
}

__device__ int dev_min(int i, int j)
{
	if(i<j)
		return i;
	else
		return j;
}



__device__ void dev_getRndNeighbor(int i, int j) 
{
  int neighbor_count = (host_pixel_neighbor * 2 + 1) * (host_pixel_neighbor * 2 + 1);
  int rnd =
      dev_rng[host_rng_idx = (host_rng_idx + 1) % RANDOM_BUFFER_SIZE] % neighbor_count;
  int start_i = i - host_pixel_neighbor;
  int start_j = j - host_pixel_neighbor;
  int area = host_pixel_neighbor * 2 + 1;
  int position_i = rnd / area;
  int position_j = rnd % area;
  int cur_i = dev_max(dev_min(start_i + position_i, host_img_size[1] - 1), 0);
  int cur_j = dev_max(dev_min(start_j + position_j, host_img_size[0] - 1), 0);
  
  rnd_pos[0]=cur_i;
  rnd_pos[1]=cur_j;
}


__global__ void init(unsigned char* dev_model, /*int channels, int samples, */
					 unsigned char* dev_image, int* dev_size, 
					 /*int pixel_neighbor,*/int* rng_idx)//处理核函数
{
	//int channels =1 ;
	//int samples =20;
	//int pixel_neighbor = 4;
	//int rng_idx = 0;
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	//cv::Size size = img.size();
	long image_size_except_channels = host_channels * dev_size[0] * dev_size[1]
										+ host_channels * dev_size[1];
	
	for (size_t k=tid; k < image_size_except_channels; k+=block_num * thread_num_per_block)
	{
		for(int c=0;c<host_channels;c++)
		{
			dev_model[k * host_samples + c] = dev_image[k + c];
		}

		int i = k/dev_size[0];
		int j = k/dev_size[1];

		for(int s=1; s<host_samples;s++)
		{	
			dev_getRndNeighbor(i, j);
			
			int img_idx = host_channels * dev_size[0] * rnd_pos[0] + host_channels * rnd_pos[1];
			int model_idx = k * host_samples + host_channels * s;
			for (int c = 0; c < host_channels; c++) 
			{
				dev_model[model_idx + c] = dev_image[img_idx + c];
			}
		}
		
	}
}

extern "C" int vibe_init_cuda(unsigned char* model, const cv::Mat &img)
{	
	
	unsigned char * dev_image;
	unsigned char * dev_model;
	//unsigned int * dev_rng;
	//int *dev_size;

	long model_size = host_channels * host_img_size[0] * host_img_size[1] * host_samples;

	//分配显卡内存
	checkCudaErrors(cudaMalloc((void**)&dev_image, sizeof(unsigned char) * 
								host_channels * host_img_size[0] *host_img_size[1] ));
	checkCudaErrors(cudaMalloc((void**)&dev_model, sizeof(unsigned char)* model_size));
	checkCudaErrors(cudaMalloc((void**)&dev_rng, sizeof(unsigned int)* RANDOM_BUFFER_SIZE));
	//checkCudaErrors(cudaMalloc((void**)&dev_size, sizeof(int)* 2));

	//将主机待处理数据内存块复制到显卡内存中
	checkCudaErrors(cudaMemcpy(dev_image, img.data, sizeof(unsigned char)* 
				host_channels * host_img_size[0] *host_img_size[1] , cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rng, host_rng, sizeof(unsigned int)* 
				RANDOM_BUFFER_SIZE, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(dev_size, img_size, sizeof(int)*2, cudaMemcpyHostToDevice));

	//调用显卡处理数据
	init<<< block_num, thread_num_per_block >>>( dev_model, dev_image, dev_size, dev_rng);

	//将显卡处理完数据拷回来
	checkCudaErrors(cudaMemcpy(model, dev_model, sizeof(unsigned char)* model_size, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(rng, dev_rng, sizeof(unsigned int)* RANDOM_BUFFER_SIZE, cudaMemcpyDeviceToHost));

	cudaFree(dev_image);//清理显卡内存
	cudaFree(dev_model);
	cudaFree(dev_rng);
	return 0;
}
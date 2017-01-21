#include "vibe.h"
#include <cuda_runtime.h> 
#include <opencv2/core/core.hpp>
#include <random>
#include <time.h>
#include <iostream>
#include <fstream>

#define thread_num_per_block 256
#define block_num 128

extern int cnt;
std::ofstream fout("D:\\cuda.txt");

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

__device__ void dev_getRndNeighbor(int i, int j, unsigned int *dev_rng, int width,
								int height, int* rng_idx, int pixel_neighbor_,
								int *rnd_pos)
{
	int neighbor_count = (pixel_neighbor_ * 2 + 1) * (pixel_neighbor_ * 2 + 1);
	*rng_idx = (*rng_idx + 1) % RANDOM_BUFFER_SIZE;
	int rnd = dev_rng[*rng_idx] % neighbor_count;
	int start_i = i - pixel_neighbor_;
	int start_j = j - pixel_neighbor_;
	int area = pixel_neighbor_ * 2 + 1;
	int position_i = rnd / area;
	int position_j = rnd % area;

	int cur_i = dev_max(dev_min(start_i + position_i, height - 1), 0);
	int cur_j = dev_max(dev_min(start_j + position_j, width - 1), 0);
	rnd_pos[0] = cur_i;
	rnd_pos[1] = cur_j;

}

__global__ void init_cuda(unsigned char* model, unsigned char* image, 
					unsigned int* rng, int width, int height, 
					int channels, int samples, 
					int pixel_neighbor, int* rng_idx)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	long image_length = width * height;

	for (size_t k=tid;k<image_length;k+=block_num*thread_num_per_block)
	{
		int i = k/width;
		int j = k%width;

		int rnd_pos[2];
		for (int c = 0; c < channels; c++) 
		{
			model[channels * samples * width * i +
					channels * samples * j + c] =
				image[channels * width * i + channels * j + c];
		}
		for (int s = 1; s < samples; s++) 
		{
			dev_getRndNeighbor(i, j, rng, width, height, rng_idx, pixel_neighbor, rnd_pos);
			int img_idx =
			    channels * width * rnd_pos[0] + channels * rnd_pos[1];
			int model_idx = channels * samples * width * i +
                        channels * samples * j + channels * s;
			for (int c = 0; c < channels; c++) 
			{
			    model[model_idx + c] = image[img_idx + c];
		    }
		}
	}
}

__global__ void update_cuda(unsigned char* model, unsigned char* image,
					unsigned char* mask, unsigned int* rng,
					int width, int height,int pixel_neighbor,
					int distance_threshold, int matching_threshold,
					int update_factor, int channels, int samples,
					int* rng_idx )
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	long image_length = width * height;

	int added_num = block_num*thread_num_per_block;
	for (size_t k=tid;k<image_length;k+=added_num)
	{
		int i = k/width;
		int j = k%width;

		bool flag = false;
		int matching_counter = 0;
		int img_idx = width * i + j;

		for(int s=0; s<samples; s++)
		{
			int model_idx = samples * (width * i + j) + s;
			if(std::abs(image[img_idx] - model[model_idx])
								< distance_threshold)
				matching_counter++;
			//背景模型与新一帧图像匹配点数超过阀值，像素点（i，j）为背景点
			if (matching_counter > matching_threshold) 
			{
				flag = true;
				break;
			}
		}
		
		if (flag) 
		{
			mask[width * i + j] = 0;//黑色
			*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
			//按照一定概率背景模型，更新因子update_factor
			if (rng[(*rng_idx)] % update_factor) 
			{
				*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
				int sample = rng[(*rng_idx)] % samples;
				int model_idx = samples *width * i +
						  samples * j + sample;
				//更新当前像素点(x,y)的背景模型
				model[model_idx] = image[img_idx];
				//在邻域中随机选取一点更新它的背景模型
				int rnd_pos[2];
				dev_getRndNeighbor(i, j, rng, width, height, rng_idx,
					pixel_neighbor, rnd_pos);
				*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
				sample = rng[(*rng_idx)] % samples;
				model_idx = samples * width * rnd_pos[0] +
					samples * rnd_pos[1] + sample;
				model[model_idx] = image[img_idx];
			}
		} else 
		{
			mask[width * i + j] = 255;//白色
		}
	}
}

namespace masa_video_compression {

VIBE::VIBE(int channels, int samples, int pixel_neighbor,
           int distance_threshold, int matching_threshold, int update_factor)
    : samples_(samples), channels_(channels), pixel_neighbor_(pixel_neighbor),
      distance_threshold_(distance_threshold),
      matching_threshold_(matching_threshold), update_factor_(update_factor) {

  model_ = nullptr;
  rng_idx_ = 0;
  srand(0);
  for (int i = 0; i < RANDOM_BUFFER_SIZE; i++) {
    rng_[i] = rand();
  }
}

VIBE::~VIBE() {
  if (model_ != nullptr) {
    delete[] model_;
  }
    cudaFree(dev_image);//清理显卡内存
	cudaFree(dev_model);
	cudaFree(dev_mask);
	cudaFree(dev_rng);
	cudaFree(dev_rng_idx);
}

void VIBE::init(const cv::Mat &img) {
  CV_Assert(img.channels() == channels_);
  size_ = img.size();
  model_ = new unsigned char[channels_ * size_.width * size_.height * samples_];
  mask_ = cv::Mat(size_, CV_8UC1, cv::Scalar::all(0));

	width = img.size().width;
	height = img.size().height;
	fout<<"video information:"<<width<<"*"<<height<<std::endl;

	image_size = width * height;
	model_size = channels_ * image_size * samples_;

	long init_start  = clock();
	//分配显卡内存
	checkCudaErrors(cudaMalloc((void**)&dev_image, sizeof(unsigned char)* 
		channels_ * image_size ));
	checkCudaErrors(cudaMalloc((void**)&dev_model, sizeof(unsigned char)* 
		model_size));
	checkCudaErrors(cudaMalloc((void**)&dev_mask, sizeof(unsigned char)* 
		channels_ * image_size ));
	checkCudaErrors(cudaMalloc((void**)&dev_rng, sizeof(unsigned int)* 
		RANDOM_BUFFER_SIZE));
	checkCudaErrors(cudaMalloc((void**)&dev_rng_idx, sizeof(unsigned int)));
	
	//将主机待处理数据内存块复制到显卡内存中
	checkCudaErrors(cudaMemcpy(dev_image, img.data, sizeof(unsigned char)* 
				channels_ * image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_mask, mask_.data, sizeof(unsigned char)* 
				image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rng, rng_, sizeof(unsigned int)* 
				RANDOM_BUFFER_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rng_idx, &rng_idx_, sizeof(unsigned int),
				cudaMemcpyHostToDevice));

	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);

	init_cuda<<<block_num, thread_num_per_block>>>//调用显卡处理数据
			(dev_model, dev_image, dev_rng, width, height, channels_, 
			samples_, pixel_neighbor_, dev_rng_idx);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);

	fout<<"init time:"<<msecTotal1<<std::endl;
}

void VIBE::update(const cv::Mat &img) {
  CV_Assert(channels_ == img.channels() && size_ == img.size());
  
	

	cudaEvent_t start1;
	cudaEventCreate(&start1);
	cudaEvent_t stop1;
	cudaEventCreate(&stop1);
	cudaEventRecord(start1, NULL);
	checkCudaErrors(cudaMemcpy(dev_image, img.data, sizeof(unsigned char)* 
				channels_ * image_size, cudaMemcpyHostToDevice));
	update_cuda<<<block_num, thread_num_per_block>>>//调用显卡处理数据
		(	dev_model, dev_image, dev_mask, dev_rng, width, height, pixel_neighbor_, 
			distance_threshold_, matching_threshold_, update_factor_, channels_, 
			samples_, dev_rng_idx);

	cudaEventRecord(stop1, NULL);
	cudaEventSynchronize(stop1);
	float msecTotal1 = 0.0f;
	cudaEventElapsedTime(&msecTotal1, start1, stop1);

	fout<<"NO."<<cnt<<" time: "<<msecTotal1<<"ms\n";
}

cv::Mat &VIBE::getMask()
{ 
	checkCudaErrors(cudaMemcpy(mask_.data, dev_mask, sizeof(unsigned char)*
		image_size, cudaMemcpyDeviceToHost));
	return mask_;
}

}

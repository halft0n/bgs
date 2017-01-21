#include <cuda_runtime.h> 
#include "vibe.h"
#include <opencv2/core/core.hpp>
#include <random>

#define thread_num_per_block 512
#define block_num 64

//cv::Mat mask_;
//unsigned char* dev_mask;
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
	//dev_rng[*rng_idx]++;
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

__global__ void init(unsigned char* model, unsigned char* image, 
					unsigned int* rng, int width, int height, 
					int channels, int samples, 
					int pixel_neighbor, int* rng_idx)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	long image_length = width * height;
	//long model_length = image_length * samples * channels;

	for (size_t k=tid;k<image_length;k+=block_num*thread_num_per_block)
	{
		int i = k/width;
		int j = k%width;
		//实验验证下面的方案不行，如果直接定义 k<model_length ，第一帧
		//图像出现的白点（运动目标）太多，效果很差，原因还待进一步研究
		//int i = k / (width * samples * channels);
		//int j = (k % (width * samples * channels))
		//	/ (samples * channels);
		//int s =(k % (samples * channels ))
		//	/ channels;
		//s++;
		//int c = k % channels;
		
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

__global__ void update(unsigned char* model, unsigned char* image,
					unsigned char* mask, unsigned int* rng,
					int width, int height,int pixel_neighbor,
					int distance_threshold, int matching_threshold,
					int update_factor, int channels, int samples,
					int* rng_idx)
{
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	long image_length = width * height;
	for (size_t k=tid;k<image_length;k+=block_num*thread_num_per_block)
	{
		int i = k/width;
		int j = k%width;

		bool flag = false;
		int matching_counter = 0;
		int img_idx = channels * width * i + channels * j;

		for(int s=0; s<samples; s++)
		{
			int model_idx = channels * samples * width * i +
							channels * samples * j + channels * s;
			int channels_counter = 0;
			for (int c = 0; c < channels; c++) 
			{
				  if (std::abs(image[img_idx + c] - model[model_idx + c]) <
					  distance_threshold) {
					channels_counter++;
				  }
			}

			if (channels_counter == channels) 
			{
				  matching_counter++;
			}
			if (matching_counter > matching_threshold) 
			{
				  flag = true;
				  break;
			}
		}

		if (flag) 
		{
			mask[width * i + j] = 0;
			*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
			if (rng[(*rng_idx)] % update_factor) 
			{
				*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
				int sample = rng[(*rng_idx)] % samples;
				int model_idx = channels * samples *width * i +
						  channels * samples * j + channels * sample;
				for (int c = 0; c < channels; c++) 
				{
					model[model_idx + c] = image[img_idx + c];
				}
				int rnd_pos[2];
				dev_getRndNeighbor(i, j, rng, width, height, rng_idx, pixel_neighbor, rnd_pos);
				*rng_idx = ((*rng_idx) + 1) % RANDOM_BUFFER_SIZE;
				sample = rng[(*rng_idx)] % samples;
				model_idx = channels * samples * width * rnd_pos[0] +
								channels * samples * rnd_pos[1] + channels * sample;
				for (int c = 0; c < channels; c++) 
				{
					model[model_idx + c] = image[img_idx + c];
				}
			}
		} else 
		{
			mask[width * i + j] = 255;
		}
	}
}


extern "C" int vibe_init_cuda(unsigned char* model, const cv::Mat &img,
						int channels_, int samples_, int pixel_neighbor_,
						unsigned int* rng_,int* rng_idx_)
{	

	unsigned char * dev_image;
	unsigned char * dev_model;
	unsigned int * dev_rng;
	int * dev_rng_idx;

	int width = img.size().width;
	int height = img.size().height;

	long image_size = width * height;
	long model_size = channels_ * image_size * samples_;

//***********************2015.04.08 added*********************************/

	//CV_Assert(img.channels() == channels_);
	//size_ = img.size();
	//unsigned char * model = new unsigned char[channels_ * width * height * samples_];
	//cv::Mat mask_ = cv::Mat(img.size(), CV_8UC1, cv::Scalar::all(0));

	//unsigned char * dev_mask;
	//checkCudaErrors(cudaMalloc((void**)&dev_mask, sizeof(unsigned char)* 
	//	channels_ * image_size ));
	//checkCudaErrors(cudaMemcpy(dev_mask, mask_.data, sizeof(unsigned char)* 
	//			image_size, cudaMemcpyHostToDevice));


//***********************************************************************************//

	//分配显卡内存
	checkCudaErrors(cudaMalloc((void**)&dev_image, sizeof(unsigned char)* 
		channels_ * image_size ));
	checkCudaErrors(cudaMalloc((void**)&dev_model, sizeof(unsigned char)* 
		model_size));
	checkCudaErrors(cudaMalloc((void**)&dev_rng, sizeof(unsigned int)* 
		RANDOM_BUFFER_SIZE));
	checkCudaErrors(cudaMalloc((void**)&dev_rng_idx, sizeof(unsigned int)));

	//将主机待处理数据内存块复制到显卡内存中
	checkCudaErrors(cudaMemcpy(dev_image, img.data, sizeof(unsigned char)* 
				channels_ * image_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rng, rng_, sizeof(unsigned int)* 
				RANDOM_BUFFER_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_rng_idx, rng_idx_, sizeof(unsigned int),
				cudaMemcpyHostToDevice));

	init<<<block_num, thread_num_per_block>>>//调用显卡处理数据
			(dev_model, dev_image, dev_rng, width, height, channels_, 
			samples_, pixel_neighbor_, dev_rng_idx);


	//将显卡处理完数据拷回来
	checkCudaErrors(cudaMemcpy(model, dev_model, sizeof(unsigned char)*
		model_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rng_idx_, dev_rng_idx, sizeof(unsigned int), 
		cudaMemcpyDeviceToHost));

	cudaFree(dev_image);//清理显卡内存
	cudaFree(dev_model);
	cudaFree(dev_rng);
	cudaFree(dev_rng_idx);
	return 0;
}

extern "C" int vibe_update_cuda(unsigned char* model, const cv::Mat &img,
						cv::Mat &mask_, int distance_threshold_,
						int matching_threshold_, int update_factor_,
						int channels_, int samples_, int pixel_neighbor_,
						unsigned int* rng_,int* rng_idx_)
{
	unsigned char * dev_image;
	unsigned char * dev_model;
	unsigned char * dev_mask;
	unsigned int * dev_rng;
	int * dev_rng_idx;

	int width = img.size().width;
	int height = img.size().height;

	long image_size = width * height;
	long model_size = channels_ * image_size * samples_;

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
	checkCudaErrors(cudaMemcpy(dev_rng_idx, rng_idx_, sizeof(unsigned int),
				cudaMemcpyHostToDevice));

	update<<<block_num, thread_num_per_block>>>//调用显卡处理数据
		(	dev_model, dev_image, dev_mask, dev_rng, width, height, pixel_neighbor_, 
			distance_threshold_, matching_threshold_, update_factor_, channels_, 
			samples_, dev_rng_idx );

	//将显卡处理完数据拷回来
	checkCudaErrors(cudaMemcpy(model, dev_model, sizeof(unsigned char)*
		model_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mask_.data, dev_mask, sizeof(unsigned char)*
		image_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rng_idx_, dev_rng_idx, sizeof(unsigned int), 
		cudaMemcpyDeviceToHost));

	cudaFree(dev_image);//清理显卡内存
	cudaFree(dev_model);
	cudaFree(dev_mask);
	cudaFree(dev_rng);
	cudaFree(dev_rng_idx);
	return 0;
	//return mask_.data;

}
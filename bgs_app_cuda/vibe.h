#ifndef VIDEO_COMPRESSION_VIBE_H
#define VIDEO_COMPRESSION_VIBE_H

#include "utils.h"
#include <opencv2/core/core.hpp>
#include <memory>

#define RANDOM_BUFFER_SIZE (65535)

namespace masa_video_compression {

class VIBE {
public:
  VIBE(int channels = 1, int samples = 20, int pixel_neighbor = 1,
       int distance_threshold = 20, int matching_threshold = 3,
       int update_factor = 16);
  ~VIBE();
  void init(const cv::Mat &img);
  void update(const cv::Mat &img);
  cv::Mat &getMask();

private:
  
  int samples_;
  int channels_;
  int pixel_neighbor_;
  int distance_threshold_;
  int matching_threshold_;
  int update_factor_;

    unsigned char * dev_image;
	unsigned char * dev_model;
	unsigned char * dev_mask;
	unsigned int * dev_rng;
	int * dev_rng_idx;

	int width;
	int height;

	long image_size;
	long model_size;

  cv::Size size_;
  unsigned char *model_;

  cv::Mat mask_;

  unsigned int rng_[RANDOM_BUFFER_SIZE];
  int rng_idx_;
  
  DISALLOW_COPY_AND_ASSIGN(VIBE);
};
}

#endif //_VIDEO_COMPRESSION_VIBE_H

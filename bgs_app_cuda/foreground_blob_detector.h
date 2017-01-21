#ifndef VIDEO_COMPRESSION_FOREGROUND_BLOB_DETECTOR_H
#define VIDEO_COMPRESSION_FOREGROUND_BLOB_DETECTOR_H
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace masa_video_compression {
class ForegroundBlobDetector {
public:
  struct Params {
    Params();
    float minDistBetweenBlobs;

    bool filterByColor;
    uchar blobColor;

    bool filterByArea;
    float minArea, maxArea;

    bool filterByCircularity;
    float minCircularity, maxCircularity;

    bool filterByInertia;
    float minInertiaRatio, maxInertiaRatio;

    bool filterByConvexity;
    float minConvexity, maxConvexity;
  };

  struct Center {
    cv::Point2d location;
    double radius;
    double confidence;
    double area;
  };

  explicit ForegroundBlobDetector(const ForegroundBlobDetector::Params &params =
                                      ForegroundBlobDetector::Params());

  void FindBlobs(const cv::Mat &image, const cv::Mat &binaryImage,
                 std::vector<Center> &centers) const;

protected:
  Params params_;

private:
  DISALLOW_COPY_AND_ASSIGN(ForegroundBlobDetector);
};
}
#endif

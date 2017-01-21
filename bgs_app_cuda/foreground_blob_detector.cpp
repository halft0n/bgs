#include "foreground_blob_detector.h"
#include <opencv2/highgui/highgui.hpp>
//#define DEBUG_BLOB_DETECTOR
namespace masa_video_compression {
ForegroundBlobDetector::Params::Params() {
  minDistBetweenBlobs = 10;

  filterByColor = true;
  blobColor = 0;

  filterByArea = true;
  minArea = 25;
  maxArea = 5000;

  filterByCircularity = false;
  minCircularity = 0.8f;
  maxCircularity = std::numeric_limits<float>::max();

  filterByInertia = true;
  // minInertiaRatio = 0.6;
  minInertiaRatio = 0.1f;
  maxInertiaRatio = std::numeric_limits<float>::max();

  filterByConvexity = true;
  // minConvexity = 0.8;
  // minConvexity = 0.95f;
  minConvexity = 0.80f;
  maxConvexity = std::numeric_limits<float>::max();
}

ForegroundBlobDetector::ForegroundBlobDetector(
    const ForegroundBlobDetector::Params &params)
    : params_(params) {}

void ForegroundBlobDetector::FindBlobs(const cv::Mat &image,
                                       const cv::Mat &binaryImage,
                                       std::vector<Center> &centers) const {
  // cv::Mat image = _image.getMat(), binaryImage = _binaryImage.getMat();
  //(void)image;
  centers.clear();

  std::vector<std::vector<cv::Point> > contours;
  cv::Mat tmpBinaryImage = binaryImage.clone();
  cv::findContours(tmpBinaryImage, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);
#ifdef DEBUG_BLOB_DETECTOR
  cv::Mat keypointsImage;
  cv::cvtColor(binaryImage, keypointsImage, CV_GRAY2RGB);

// cv::Mat contoursImage;
// cv::cvtColor( binaryImage, contoursImage, CV_GRAY2RGB);
// cv::drawContours( contoursImage, contours, -1, cv::Scalar(0,255,0) );
// cv::imshow("contours", contoursImage );
#endif
  // std::numeric_limits<double>::epsilon();
  for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
    Center center;
    center.confidence = 1;
    cv::Moments moms = cv::moments(cv::Mat(contours[contourIdx]));
    if (params_.filterByArea) {
      double area = moms.m00;
      if (area < params_.minArea || area >= params_.maxArea)
        continue;
    }
    center.area = moms.m00;

    if (params_.filterByCircularity) {
      double area = moms.m00;
      double perimeter = cv::arcLength(cv::Mat(contours[contourIdx]), true);
      double ratio = 4 * CV_PI * area / (perimeter * perimeter);
      if (ratio < params_.minCircularity || ratio >= params_.maxCircularity)
        continue;
    }

    if (params_.filterByInertia) {
      double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) +
                                     std::pow(moms.mu20 - moms.mu02, 2));
      const double eps = 1e-2;
      double ratio;
      if (denominator > eps) {
        double cosmin = (moms.mu20 - moms.mu02) / denominator;
        double sinmin = 2 * moms.mu11 / denominator;
        double cosmax = -cosmin;
        double sinmax = -sinmin;

        double imin = 0.5 * (moms.mu20 + moms.mu02) -
                      0.5 * (moms.mu20 - moms.mu02) * cosmin -
                      moms.mu11 * sinmin;
        double imax = 0.5 * (moms.mu20 + moms.mu02) -
                      0.5 * (moms.mu20 - moms.mu02) * cosmax -
                      moms.mu11 * sinmax;
        ratio = imin / imax;
      } else {
        ratio = 1;
      }

      if (ratio < params_.minInertiaRatio || ratio >= params_.maxInertiaRatio)
        continue;

      center.confidence = ratio * ratio;
    }

    if (params_.filterByConvexity) {
      std::vector<cv::Point> hull;
      cv::convexHull(cv::Mat(contours[contourIdx]), hull);
      double area = cv::contourArea(cv::Mat(contours[contourIdx]));
      double hullArea = cv::contourArea(cv::Mat(hull));
      double ratio = area / hullArea;
      if (ratio < params_.minConvexity || ratio >= params_.maxConvexity)
        continue;
    }

    center.location = cv::Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

    if (params_.filterByColor) {
      if (binaryImage.at<uchar>(cvRound(center.location.y),
                                cvRound(center.location.x)) !=
          params_.blobColor)
        continue;
    }
    // compute blob radius
    {
      std::vector<double> dists;
      for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size();
           pointIdx++) {
        cv::Point2d pt = contours[contourIdx][pointIdx];
        dists.push_back(norm(center.location - pt));
      }
      std::sort(dists.begin(), dists.end());
      center.radius =
          (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
    }

    if (moms.m00 == 0.0)
      continue;
    centers.push_back(center);

#ifdef DEBUG_BLOB_DETECTOR
    cv::circle(keypointsImage, center.location, center.radius,
               cv::Scalar(0, 0, 255), 1);
#endif
  }
#ifdef DEBUG_BLOB_DETECTOR
  cv::imshow("bk", keypointsImage);
  cv::waitKey(1);
#endif
}
}

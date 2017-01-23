// bgs_app.cpp : 定义控制台应用程序的入口点。
//

#include "vibe.h"
//#include "foreground_blob_detector.h"
#include <iostream>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>

int main(int argc, char **argv) {
  if (argc < 2) {
	std::cout<<"Usage: bgs_app <video file name>\n";
	return 1;
  }
  std::string filename = std::string(argv[1]);
  cv::VideoCapture cap(filename);
  if (!cap.isOpened())
    return 1;

  filename.insert(filename.rfind('.'), "_fg");
  printf("%s\n", filename.c_str());
  cv::VideoWriter vwriter(filename, CV_FOURCC('X', 'V', 'I', 'D'),
                          cap.get(CV_CAP_PROP_FPS),
                          cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
                                   cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

  int channels = 1;
  //masa_video_compression::VIBE vibe_bgs(channels, 20, 4, 17, 2, 16);
  masa_video_compression::VIBE vibe_bgs(channels,30,1,25,3,16);
  cv::Mat frame;
  cap >> frame;
  if (frame.empty()) {
    printf("Failed to read frame!\n");
    return 1;
  }

  //if (channels == 1) {
  //  cv::cvtColor(frame, frame, CV_BGR2GRAY);
  //} else {
  //  cv::cvtColor(frame, frame, CV_BGR2HSV);
  //}
  cv::Mat gray;
  cv::cvtColor(frame, gray, CV_BGR2GRAY);

  // vibe_bgs.init(frame);
  vibe_bgs.init(gray);
  
  // cv::Mat gray = frame.clone();
  while (!frame.empty()) {
    cv::cvtColor(frame, gray, CV_BGR2GRAY);
	
    vibe_bgs.update(gray);
	cv::Mat foreground = vibe_bgs.getMask();

    // frame.copyTo(foreground, vibe_bgs.getMask());
    cv::Mat vout;
    cv::cvtColor(foreground, vout, CV_GRAY2BGR);
    vwriter << vout; 

    cv::imshow("frame", frame);
    cv::imshow("foreground", foreground);
    cap >> frame;

    char key = cv::waitKey(1);
    if (key == 27) {
      break;
    }
  }
  vwriter.release();
  printf("Release video writer!\n");
  return 0;
}


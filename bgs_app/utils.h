#ifndef VIDEO_COMPRESSION_UTILS_H
#define VIDEO_COMPRESSION_UTILS_H
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#define DISALLOW_COPY_AND_ASSIGN(TypeName)                                     \
  TypeName(const TypeName &);                                                  \
  void operator=(const TypeName &)

namespace masa_video_compression {
bool GetActiveClip(const std::string &max_area_file_name, const float area_thd,
                   const int length_thd, const int dist_thd,
                   std::vector<int> &start_pos, std::vector<int> &end_pos);
}
#endif

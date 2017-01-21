#include "utils.h"
#include <stdio.h>
#include <string>

namespace masa_video_compression {
bool GetActiveClip(const std::string &max_area_file_name, const float area_thd,
                   const int length_thd, const int dist_thd,
                   std::vector<int> &start_pos, std::vector<int> &end_pos) {
//----------------altered by wyp---------------//
  //FILE *max_area_file = fopen(max_area_file_name.c_str(), "r");
  errno_t err_fopen;

  FILE *max_area_file;
  if((err_fopen = fopen_s(&max_area_file,max_area_file_name.c_str(), "r"))!=0)
  {
	  printf("open file failed!\n");
	  return false;
  }
//----------------alter end-----------------------//
  if (!max_area_file)
    return false;
  std::vector<float> max_area_arr;
  max_area_arr.push_back(0);
  while (!feof(max_area_file)) {
    float area;
    fscanf(max_area_file, "%f", &area);
    max_area_arr.push_back(area);
  }
  max_area_arr.push_back(0);
  // printf("Num frames %lu\n", max_area_arr.size());

  start_pos.clear();
  end_pos.clear();
  for (int i = 1; i < max_area_arr.size() - 1; i++) {
    if (max_area_arr[i - 1] < area_thd && max_area_arr[i] > area_thd)
      start_pos.push_back(i - 1);
    if (max_area_arr[i] > area_thd && max_area_arr[i + 1] < area_thd)
      end_pos.push_back(i - 1);
  }
  // printf("Original size: %lu %lu\n", start_pos.size(), end_pos.size());
  // for (int i = 0; i < start_pos.size(); i++) {
  // printf("%d %d\n", start_pos[i], end_pos[i]);
  //}

  std::vector<int>::iterator start_iter = start_pos.begin();
  std::vector<int>::iterator end_iter = end_pos.begin();
  for (; start_iter != start_pos.end();) {
    if (*end_iter - *start_iter < length_thd) {
      start_iter = start_pos.erase(start_iter);
      end_iter = end_pos.erase(end_iter);
    } else {
      start_iter++;
      end_iter++;
    }
  }
  // printf("After length filter: %lu %lu\n", start_pos.size(), end_pos.size());
  // for (int i = 0; i < start_pos.size(); i++) {
  // printf("%d %d\n", start_pos[i], end_pos[i]);
  //}

  start_iter = start_pos.begin();
  end_iter = end_pos.begin();
  for (; start_iter != start_pos.end() - 1;) {
    if (*(start_iter + 1) - *end_iter < dist_thd) {
      start_pos.erase(start_iter + 1);
      end_iter = end_pos.erase(end_iter);
    } else {
      start_iter++;
      end_iter++;
    }
  }
  // printf("After close clips mergence: %lu %lu\n", start_pos.size(),
  // end_pos.size());
  // for (int i = 0; i < start_pos.size(); i++) {
  // printf("%d %d\n", start_pos[i], end_pos[i]);
  //}

  fclose(max_area_file);
  return true;
}
}

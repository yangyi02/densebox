 
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
 
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"


using namespace caffe;

int main(int argc, char** argv) {
  if (argc < 3) {
    LOG(ERROR)<< "FDDB2KITTI.bin"<<
    			" FDDB_result  "<<
    			" KITTI_result_fold  ";

    return 0;
  }
  char img_name[256];
  FILE* pred_fd = NULL;
  FILE* FG_pred_fd = NULL;
  pred_fd  = fopen(argv[1],"r");

//  vector<float> thresholds;
//  vector<int>   multiple_times;
//
//  thresholds.push_back(1.6);
//  multiple_times.push_back(4);
//
//  thresholds.push_back(1.2);
//  multiple_times.push_back(2);
//
//  thresholds.push_back(1.2);
//  multiple_times.push_back(1);

  int file_count =0;
  while(fscanf(pred_fd,"%s",img_name) == 1)
  {
	  string file_name(argv[2]);
	  file_name = file_name +string("/")+ string(img_name)+ string(".txt");
	  FG_pred_fd = fopen(file_name.c_str(),"w");
	  file_count++;
	  int n_face = 0;
	  CHECK(fscanf(pred_fd,"%d",&n_face) == 1);
	  for(int i=0; i < n_face; i++)
	  {
		  float lt_x, lt_y, height,width,score;
		  CHECK(fscanf(pred_fd, "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);

		  fprintf(FG_pred_fd,"%s 0 0 0 %f %f %f %f 0 0 0 0 0 0 0 %f \n","Car",
				   (lt_x), (lt_y),  (lt_x+width), (lt_y+height),score);

		  /**
		   * double high score candidates
		   */
//		  for(int t=0; t < thresholds.size();++t){
//			  if(score >= thresholds[t]){
//				  for(int j=0; j<multiple_times[t];++j){
//					  fprintf(FG_pred_fd,"%s 0 0 0 %f %f %f %f 0 0 0 0 0 0 0 %f \n","Car",
//								   (lt_x), (lt_y),  (lt_x+width), (lt_y+height),score);
//				  }
//			  }
//		  }
	  }
	  fclose(FG_pred_fd);
  }
  fclose(pred_fd);
  std::cout<<"file_cout: "<<file_count<<std::endl;
  return 0;
}

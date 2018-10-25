
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
    LOG(ERROR)<< "FDDB2FG.bin"<<
    			" FDDB_result  "<<
    			" FG_result  ";

    return 0;
  }
  char img_name[256];
  FILE* pred_fd = NULL;
  FILE* FG_pred_fd = NULL;
  pred_fd  = fopen(argv[1],"r");
  FG_pred_fd = fopen(argv[2],"w");
  int file_count =0;
  while(fscanf(pred_fd,"%s",img_name) == 1)
  {
	  file_count++;
	  int n_face = 0;
	  CHECK(fscanf(pred_fd,"%d",&n_face) == 1);
	  for(int i=0; i < n_face; i++)
	  {
		  float lt_x, lt_y, height,width,score;
		  CHECK(fscanf(pred_fd, "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
		  fprintf(FG_pred_fd,"%s %d %d %d %d %f \n",img_name,
				  int(lt_x),int(lt_y), int(width),int(height),score);
	  }
  }
  fclose(pred_fd);
  fclose(FG_pred_fd);
  std::cout<<"file_cout: "<<file_count;
  return 0;
}

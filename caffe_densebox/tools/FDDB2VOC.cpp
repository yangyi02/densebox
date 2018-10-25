
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
  if (argc < 5) {
    LOG(ERROR)<< "FDDB2VOC.bin"<<
    			" class_list  input_folder FDDB_result_postfix  "<<
    			" VOC_result_prefix  ";

    return 0;
  }

	vector<string> class_names_;
	string file_list_name = string(argv[1]);
	LOG(INFO) << "Opening class list file " << file_list_name;
	std::ifstream infile(file_list_name.c_str());
	CHECK(infile.good());
	string class_name;
	while(std::getline(infile,class_name)){
		if(class_name.empty()) continue;
		class_names_.push_back(class_name);
	}
	infile.close();
	for(int i=0; i <class_names_.size(); ++i){
	  char img_name[256];
	  FILE* pred_fd = NULL;
	  FILE* FG_pred_fd = NULL;
	  string pred_filename = string(argv[2])+ class_names_[i] + string(argv[3]);
	  string out_filename = string(argv[4]) + class_names_[i] + string(".txt");
	  pred_fd  = fopen(pred_filename.c_str(),"r");
	  FG_pred_fd = fopen(out_filename.c_str(),"w");
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
			  fprintf(FG_pred_fd,"%s %f %d %d %d %d\n",img_name,score,
					  int(lt_x),int(lt_y), int(width+lt_x),int(height+lt_y));
		  }
	  }
	  fclose(pred_fd);
	  fclose(FG_pred_fd);
	  std::cout<<"file_cout: "<<file_count<<std::endl;
	}
  return 0;
}


#include <sys/stat.h>
#include <cstdio>
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
	//std::cout<<"before check"<<std::endl;
  if (argc < 4) {
    LOG(ERROR)<< "BBox_voting_FDDB.bin"<<
    			" candidate_file_list  "<<
    			" output_file "<<
    			"threshold ";
    return 0;
  }

 // std::cout<<"after check"<<std::endl;
  vector<string> file_names;
  char file_name[256];
  FILE* file_list_fid = NULL;
  FILE* out_fid = NULL;
 // std::cout<<"before vector"<<std::endl;
  vector<FILE*> file_fids;
//  std::cout<<"after vector"<<std::endl;
  float threshold = atof(argv[3]);
////  std::cout<<"before fopen"<<std::endl;
  out_fid = fopen(argv[2],"w");
  file_list_fid = fopen(argv[1],"r");
//  std::cout<<"before check"<<std::endl;
  CHECK(file_list_fid != NULL)<< "could not find file: "<<argv[1];
//  std::cout<<"after check"<<std::endl;
  while(fscanf(file_list_fid, "%s",file_name) == 1)
  {
	  FILE* temp_fid = fopen(file_name,"r");
	  CHECK(temp_fid != NULL)<< "could not fild file: "<<file_name;
	  file_fids.push_back(temp_fid);
  }
  CHECK(file_fids.size() >0 );
  LOG(INFO)<<"Merging "<<file_fids.size()<<" files";


  vector< vector<float> > bboxs;
  vector< vector<float> > result_bboxs;
  vector<float> temp_instance;
  char file_name2[256];
  while(fscanf(file_fids[0],"%s",file_name) == 1)
  {
	  // get faces candidate in the first file
	  bboxs.clear();
	  result_bboxs.clear();
	  int n_face = 0;
	  CHECK(fscanf(file_fids[0],"%d",&n_face) == 1);
	  float lt_x, lt_y, height,width,score;
	  for(int i_face = 0; i_face < n_face; i_face ++)
	  {
		  CHECK(fscanf(file_fids[0], "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
		  temp_instance.clear();
		  temp_instance.push_back(score);
		  temp_instance.push_back(lt_x);
		  temp_instance.push_back(lt_y);
		  temp_instance.push_back(lt_x+width);
		  temp_instance.push_back(lt_y+height);
		  bboxs.push_back(temp_instance);
	  }
	  // get the remaining faces in other files
	  for(int file_id = 1; file_id < file_fids.size(); ++ file_id)
	  {
		  CHECK(fscanf(file_fids[file_id],"%s",file_name2) == 1);
		  CHECK(strcmp(file_name2,file_name) == 0)<<"file name: "<< file_name2 <<
				  " is different with file name: "<<file_name;
		  CHECK(fscanf(file_fids[file_id],"%d",&n_face) == 1);

		  for(int i_face = 0; i_face < n_face; i_face ++)
		  {
			  CHECK(fscanf(file_fids[file_id], "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
			  temp_instance.clear();
			  temp_instance.push_back(score);
			  temp_instance.push_back(lt_x);
			  temp_instance.push_back(lt_y);
			  temp_instance.push_back(lt_x+width);
			  temp_instance.push_back(lt_y+height);
			  bboxs.push_back(temp_instance);
		  }
	  }
	  vector<bool> selected = caffe::bbox_voting(bboxs,threshold);
	  for(int i=0 ; i < selected.size(); ++ i)
	  {
		  if(selected[i] == true)
		  {
			  result_bboxs.push_back(bboxs[i]);
		  }
	  }
	  bboxs = result_bboxs;
	  result_bboxs.clear();
	  selected = caffe::nms(bboxs,float(0.8),100,false);
	  for(int i=0 ; i < selected.size(); ++ i)
	  {
		  if(selected[i] == true)
		  {
			  result_bboxs.push_back(bboxs[i]);
		  }
	  }

	  fprintf(out_fid,"%s\n",file_name);
	  fprintf(out_fid,"%d\n",int(result_bboxs.size()));
	  for (int i=0; i < result_bboxs.size(); ++i)
	  {
		  fprintf(out_fid,"%f %f %f %f %f \n",result_bboxs[i][1],result_bboxs[i][2],
				  result_bboxs[i][3]-result_bboxs[i][1], result_bboxs[i][4]- result_bboxs[i][2],
				  result_bboxs[i][0]);
	  }
  }



  // close all file
  for(int i=0; i<file_fids.size();++i )
  {
	  fclose(file_fids[i]);
  }
  fclose(file_list_fid);
  fclose(out_fid);
  return 0;
}

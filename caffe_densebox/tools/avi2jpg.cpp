
#include <sys/stat.h>

#include <vector>

#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>


#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/buffered_reader.hpp"

using namespace std;
using namespace caffe;
int main(int argc, char** argv) {

	 if (argc < 5) {
	    LOG(ERROR)<< "avi2jpg.bin"<<
	    			" avi_file_name  "<<
	    			" output_folder "<<
	    			" frame_start_id "<<
	    			" frame_end_id "<<
	    			" frame_step";
	    return 0;
	 }
	string avi_file_name = string(argv[1]);
	string output_folder = string(argv[2]);
	int frame_start_id = atoi(argv[3]);
	int frame_end_id = atoi(argv[4]);
	int frame_step = 1;
	if(argc > 5){
		frame_step = atoi(argv[5]);
	}

	std::vector<std::string> splited_name= std_split(avi_file_name,"/");
	splited_name = std_split(splited_name[splited_name.size()-1],".avi");
	string real_avi_filename = splited_name[0];
	std::cout<<"Converting "<<avi_file_name<<" to jpg, start from frame "<< frame_start_id<<
			" to frame "<<frame_end_id <<" with step = "<<frame_step;
	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	char dst_path[512];

	cv::VideoCapture capture(avi_file_name);
	if(!capture.isOpened()){
		LOG(INFO)<<"Failed to open video "<< avi_file_name;
		return 0;
	}
	int totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	int frameToStart = frame_start_id;
	if(frameToStart >= totalFrameNumber){
		LOG(INFO)<<"Frame_id "<< frameToStart <<" exceeds total frame number "<< totalFrameNumber;
		capture.release();
		return 0;
	}

	int count = 0;
	capture.set( CV_CAP_PROP_POS_FRAMES,frame_start_id);
	for(int i=frame_start_id; i < frame_end_id; i += 1 ){

		if(i >= totalFrameNumber){
			LOG(INFO)<<"Frame_id "<< i <<" exceeds total frame number "<< totalFrameNumber;
			return 0;
		}

		
		cv::Mat dst_mat;
		if(!capture.read(dst_mat)){
			capture.release();
			LOG(INFO)<<"Failed to load frame_id "<< i <<" total frame_num "<<totalFrameNumber;
			return 0;
		}
		if((i-frame_end_id)% frame_step == 0){
			sprintf(dst_path,"%s/%s_%08d.jpg",output_folder.c_str(),real_avi_filename.c_str(),i);	
			imwrite(dst_path,dst_mat);
			count++;
			if(count % 10 == 0){
				cout<<count<<"/"<<(frame_end_id - frame_start_id)/frame_step<<" are finished"<<std::endl;
			}
		}
	}
	capture.release();
	return 0;
}

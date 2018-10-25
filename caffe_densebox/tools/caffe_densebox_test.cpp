#include <sys/stat.h>

#include <vector>

#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/pyramid_data_layers.hpp"
#include "caffe/caffe_wrapper.hpp"
using namespace caffe;
using std::vector;
using std::string;


int main(int argc, char** argv) {
	if (argc < 10) {
		LOG(ERROR)<< "test_net net_proto_prefix  "
		<< "outputfolder [CPU/GPU] [device_id] "
		<< "start_iter  end_iter  iter_step groundtruth_file show_img[0/1] ";
		return 0;
	}
	bool use_cuda = false;
	if (argc > 4 && strcmp(argv[4], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);

	} else {
		Caffe::set_mode(Caffe::GPU);
		use_cuda = true;
		int device_id = 0;
		if (argc > 5) {
			device_id = atoi(argv[5]);
		}
		Caffe::SetDevice(device_id);
		LOG(INFO) << "Using GPU #" << device_id;
	}
	int model_iter_start = atoi(argv[6]);
	int model_iter_end = atoi(argv[7]);
	int model_iter_step = atoi(argv[8]);
	string gt_file_list = string(argv[9]);
	string net_prefix = string(argv[2]);

	bool show_result = true;
	if(argc > 10){
		show_result = ( strcmp(argv[10], "1") == 0);
	}
	bool show_time =  false;
	if(argc > 11){
		show_time = ( strcmp(argv[11], "1") == 0);
	}

	string output_folder(argv[3]);
	output_folder.append("/");
	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


	LOG(INFO) << "Creating testing net...";

	CaffeDenseBoxDetector detector(argv[1],use_cuda);

	int class_num = detector.ClassNum();
	vector<float> show_threshold = vector<float>(class_num,0);


	char net_param_c[256];

	for(int net_iter = model_iter_start; net_iter < model_iter_end; net_iter+=model_iter_step){

		sprintf(net_param_c, "%s_iter_%d.caffemodel",net_prefix.c_str(),net_iter);
		LOG(INFO) << "Copy layers from trained net "<<net_param_c;
//		caffe_test_net.CopyTrainedLayersFrom(net_param_c);
		detector.CopyFromModel(net_param_c);
		LOG(INFO) << "Copy Net finished...";

		vector<string> predicted_name_list;
		vector<std::ofstream *>  out_result_files;


		string img_folder;
		img_folder.clear();

		pair<string, vector<float> > cur_sample = pair<string, vector<float> >(
				"./dataset/new_KITTI/train_jpg/000008",vector<float>());

		string cur_img_name = "./dataset/new_KITTI/train_jpg/000008.jpg";
		int cv_read_flag = CV_LOAD_IMAGE_COLOR;
		cv::Mat cv_img = cv::imread(cur_img_name, cv_read_flag);
		detector.LoadImgToBuffer(cv_img);

		detector.PredictOneImg();


			if(img_folder.empty()){
				img_folder = ImageDataSourceProvider<float>::GetSampleFolder(cur_sample) ;
				img_folder.append("/");
			}

			string img_name = ImageDataSourceProvider<float>::GetSampleName(cur_sample);

			/**
			 * ShowResult.
			 */
			if(show_result){
				string out_name = output_folder + img_name+string("_BBoxCandiates");
				ShowMultiClassBBoxOnImage(cur_sample.first,detector.GetBBoxResults(),
						show_threshold,out_name,2);
			}

	}
	return 0;
}

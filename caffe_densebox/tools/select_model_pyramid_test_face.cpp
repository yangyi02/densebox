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

	if (argc > 4 && strcmp(argv[4], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);

	} else {
		Caffe::set_mode(Caffe::GPU);
		int device_id = 0;
		if (argc > 5) {
			device_id = atoi(argv[5]);
		}
		Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
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

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);

	LOG(INFO) << "Creating testing net...";
	Net<float> caffe_test_net(argv[1],caffe::TEST);
	PyramidImageDataLayer<float>* input_data_layer;
	DetectionOutputLayer<float>* output_data_layer;
	input_data_layer = static_cast<PyramidImageDataLayer<float>*>(caffe_test_net.layers()[0].get());
	output_data_layer = static_cast<DetectionOutputLayer<float>*>(caffe_test_net.layers()[caffe_test_net.layers().size()-1].get());
	int sample_count =  input_data_layer->GetTotalSampleSize();
	char net_param_c[256];
	int class_num = output_data_layer->GetNumClass();
	vector<string> class_names = output_data_layer->GetClassNames();
	vector<float> show_threshold = vector<float>(class_num,0);
	int layer_size = caffe_test_net.layers().size();
	vector<float> forward_times(layer_size ,0);
	int show_time_interval =20;
	Timer layer_timer;

	// get gt_file from gt_file_list
	vector<string> gt_files;
	LOG(INFO) << "Opening gt_file_list: " << gt_file_list;
	std::ifstream infile(gt_file_list.c_str());
	CHECK(infile.good());
	string gt_file_name;
	while(std::getline(infile,gt_file_name)){
		if(gt_file_name.empty()) continue;
		gt_files.push_back(gt_file_name);
		LOG(INFO)<<"Get gr_file: "<<gt_file_name;
	}
	infile.close();
	CHECK_EQ(gt_files.size(), class_num);
	float sum_time = 0;

	for(int net_iter = model_iter_start; net_iter < model_iter_end; net_iter+=model_iter_step){

		sprintf(net_param_c, "%s_iter_%d.caffemodel",net_prefix.c_str(),net_iter);
		LOG(INFO) << "Copy layers from trained net "<<net_param_c;
		caffe_test_net.CopyTrainedLayersFrom(net_param_c);
		LOG(INFO) << "Copy Net finished...";

		vector<string> predicted_name_list;
		vector<std::ofstream *>  out_result_files;
		for(int class_id = 0; class_id < class_num; ++class_id){

			out_result_files.push_back(new std::ofstream());
			sprintf(net_param_c, "%s%sResult_%d.txt",output_folder.c_str(),class_names[class_id].c_str(),net_iter);
			predicted_name_list.push_back(string(net_param_c));
			out_result_files[class_id]->open(net_param_c);
			LOG(INFO)<<"test result path: "<<net_param_c;
		}

		string img_folder;
		img_folder.clear();
		LOG(INFO)<<"total number of image to be tested: "<<sample_count;
		for(int sample_id = 0; sample_id < sample_count; ++sample_id){
			if(sample_id > 0 && sample_id%show_time_interval == 0 &&show_time ){
				for(int i= 0; i < forward_times.size();++i){
					LOG(INFO)<<"Forward time for layer "<<i<<" "<<caffe_test_net.layers()[i]->layer_param().name()<<
							" :"<<forward_times[i]/show_time_interval;
					sum_time += forward_times[i]/show_time_interval;
					forward_times[i] = 0;
				}
				LOG(INFO)<<"sum_time: "<<sum_time;
				sum_time = 0;
			}


			Timer forward_timer;
			forward_timer.Start();

			if(!show_time){
				caffe_test_net.ForwardFrom(0);
			}else{
				for(int i = 0; i < layer_size  ; ++i){
					layer_timer.Start();
					caffe_test_net.ForwardFromTo(i,i );
					forward_times[i] += layer_timer.MilliSeconds();
				}
			}

			int forward_time_for_this_sample = input_data_layer->GetForwardTimesForCurSample();
			pair<string, vector<float> > cur_sample = input_data_layer->GetCurSample();

			LOG(INFO)<<"number of forward needed for cur sample: "<<forward_time_for_this_sample;
			for(int iter = 1; iter<forward_time_for_this_sample; ++iter ){
				if(!show_time){
					caffe_test_net.ForwardFrom(0);
				}else{
					for(int i = 0; i < layer_size  ; ++i){
						layer_timer.Start();
						caffe_test_net.ForwardFromTo(i,i );
						forward_times[i] += layer_timer.MilliSeconds();
					}
				}
			}

			LOG(INFO) << "Finish forwarding sample " << sample_id <<" in "<<
					forward_timer.MilliSeconds() << " ms, "<< "total sample_num: " << sample_count;

			if(img_folder.empty()){
				img_folder = ImageDataSourceProvider<float>::GetSampleFolder(cur_sample) ;
				img_folder.append("/");
			}

			string img_name = ImageDataSourceProvider<float>::GetSampleName(cur_sample);

			for(int class_id = 0; class_id < class_num; ++class_id){
				vector< BBox<float> >& result_bbox = output_data_layer->GetFilteredBBox(class_id);
				*(out_result_files[class_id]) << img_name << std::endl << result_bbox.size()<<std::endl;
				PushBBoxTo(*(out_result_files[class_id]),result_bbox);
			}

			/**
			 * ShowResult.
			 */
			if(show_result){
				string out_name = output_folder + img_name+string("_BBoxCandiates");
				ShowMultiClassBBoxOnImage(cur_sample.first,output_data_layer->GetFilteredBBox(),
						show_threshold,out_name,2);
			}
		}
		for(int class_id = 0; class_id < class_num; ++class_id){
			out_result_files[class_id]->close();
			delete out_result_files[class_id];
		}

		/**
		 *  show PR
		 */
		int n_positive=0;
		vector< std::pair<float, vector<float> > > pred_instances_with_gt;
		vector<float> precision, recall;
		float overlap = 0.5;
		std::ofstream pr_file;

		for(int class_id = 0; class_id < class_num; ++class_id){
			overlap = 0.3;
			GetPredictedWithGT_FDDB(gt_files[class_id], predicted_name_list[class_id],
				 pred_instances_with_gt,n_positive,  true,img_folder,output_folder,overlap);
			LOG(INFO)<<"Iteration "<<net_iter<<" AUC(0.3): "<< GetTPFPPoint_FDDB(pred_instances_with_gt,n_positive, precision, recall)<< "  n_instance = "<<
						   n_positive;
			sprintf(net_param_c,"%s%sAUC03_%d.txt",output_folder.c_str(),class_names[class_id].c_str(),net_iter);
			pr_file.open(net_param_c);
			for (int i=0; i< precision.size(); i++) {
				pr_file << precision[i] << " " << recall[i]<< std::endl;
			}
			pr_file.close();

			overlap = 0.5;
			GetPredictedWithGT_FDDB(gt_files[class_id], predicted_name_list[class_id],
				 pred_instances_with_gt,n_positive,  true,img_folder,output_folder,overlap);
			LOG(INFO)<<"Iteration "<<net_iter<<" AUC(0.5): "<< GetTPFPPoint_FDDB(pred_instances_with_gt,n_positive, precision, recall)<< "  n_instance = "<<
						   n_positive;
			sprintf(net_param_c,"%s%sAUC05_%d.txt",output_folder.c_str(),class_names[class_id].c_str(),net_iter);
			pr_file.open(net_param_c);
			for (int i=0; i< precision.size(); i++) {
				pr_file << precision[i] << " " << recall[i]<< std::endl;
			}
			pr_file.close();
		}

	}
	return 0;
}

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
	if (argc < 4) {
		LOG(ERROR)<< "test_net net_proto  "
		<< "outputfolder [CPU/GPU] [device_id] show_img[0/1] output_name show_time[0/1]";
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
	bool show_result = true;
	if(argc > 6){
		show_result = ( strcmp(argv[6], "1") == 0);
	}
	string output_folder(argv[3]);
	output_folder.append("/");
	string result_name("result.txt");
	if(argc > 7 ){
		result_name =  string(argv[7]);
	}

	bool show_time =  false;
	if(argc > 8){
		show_time = ( strcmp(argv[8], "1") == 0);
	}

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);

	LOG(INFO) << "Creating testing net...";
	Net<float> caffe_test_net(argv[1],caffe::TEST);
	LOG(INFO) << "Copy layers from trained net...";
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);
	LOG(INFO) << "Copy Net finished...";
	PyramidImageDataLayer<float>* input_data_layer;
	DetectionOutputLayer<float>* output_data_layer;

	input_data_layer = static_cast<PyramidImageDataLayer<float>*>(caffe_test_net.layers()[0].get());
	output_data_layer = static_cast<DetectionOutputLayer<float>*>(caffe_test_net.layers()[caffe_test_net.layers().size()-1].get());
	int sample_count =  input_data_layer->GetTotalSampleSize();
	int class_num = output_data_layer->GetNumClass();
	vector<string> class_names = output_data_layer->GetClassNames();



	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	// open file list for results.
	vector<std::ofstream *>  out_result_files;
	for(int class_id = 0; class_id < class_num; ++class_id){

		out_result_files.push_back(new std::ofstream());
		string class_result_name =  output_folder+ class_names[class_id]+ string("_") + result_name;
		out_result_files[class_id]->open(class_result_name.c_str());
		LOG(INFO)<<"test result path: "<<class_result_name;
	}
	LOG(INFO)<<"total number of image to be tested: "<<sample_count;

	int layer_size = caffe_test_net.layers().size();
	vector<float> forward_times(layer_size ,0);
	int show_time_interval =20;
	Timer layer_timer;
	vector<float> show_threshold = vector<float>(class_num,0);

	for(int sample_id = 0; sample_id < sample_count; ++sample_id){

		if(sample_id > 0 && sample_id%show_time_interval == 0 &&show_time ){
			for(int i= 0; i < forward_times.size();++i){
				LOG(INFO)<<"Forward time for layer "<<i<<" "<<caffe_test_net.layers()[i]->layer_param().name()<<
						" :"<<forward_times[i]/show_time_interval;
				forward_times[i] = 0;
			}
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
	out_result_files.clear();
	return 0;
}

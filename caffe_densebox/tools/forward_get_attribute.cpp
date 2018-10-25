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
#include "caffe/layers/landmark_detection_layers.hpp"
using namespace caffe;
using std::vector;
using std::string;




int main(int argc, char** argv) {
	if (argc < 4) {
		LOG(ERROR)<< "net_proto net_model  "
		<< "outputfolder [CPU/GPU] [device_id] show_img[0/1] output_name ";
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



	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);

	LOG(INFO) << "Creating testing net...";
	Net<float> caffe_test_net(argv[1],caffe::TEST);
	LOG(INFO) << "Copy layers from trained net...";
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);
	LOG(INFO) << "Copy Net finished...";
	LandmarkDetectionDataLayer<float>* input_data_layer;
//	LandmarkDetectionOutputLayer<float>* output_data_layer;

	input_data_layer = static_cast<LandmarkDetectionDataLayer<float>*>(caffe_test_net.layers()[0].get());
	Blob<float>* output_blob = caffe_test_net.output_blobs()[0];


//	output_data_layer = static_cast<LandmarkDetectionOutputLayer<float>*>(caffe_test_net.layers()[caffe_test_net.layers().size()-1].get());
	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	int sample_num =  input_data_layer->GetTotalSampleSize();
	int batch_size = input_data_layer->GetBatchSize();
	int total_iter_times = std::ceil((sample_num+0.0)/batch_size);
	std::ofstream   out_result_files;
	string class_result_name =  output_folder+  result_name;
	out_result_files.open(class_result_name.c_str());

	for(int iter = 0; iter < total_iter_times; ++iter){
//		std::cout<<"iter "<<iter<<" start"<< std::endl;
		caffe_test_net.ForwardFrom(0);
		int output_dim_spatial = output_blob->count(1);
		const float* output_data = output_blob->cpu_data();
		for(int sample_id = 0; sample_id < batch_size; ++sample_id){

			if(iter * batch_size + sample_id >= sample_num)
				continue;
			pair< string, vector<float> > cur_sample = input_data_layer->batch_samples()[sample_id] ;
			string img_name = ImageDataSourceProvider<float>::GetSampleName(cur_sample,1);


			std::vector<std::string> splited_name= std_split(cur_sample.first,"/");
			string out_name="";
			int split_start_id = 6;
			out_name = splited_name[split_start_id];
			for(int i=split_start_id+1; i < splited_name.size(); ++i){
					out_name = out_name + "/"+ splited_name[i] ;
			}
			img_name = out_name;

			out_result_files<< img_name<<" ";
			for(int output_id = 0; output_id < output_dim_spatial; ++ output_id){
				out_result_files<< output_data[output_dim_spatial * sample_id + output_id]<<" ";
			}
			out_result_files<<std::endl;
		}
		std::cout<<iter<<"/"<<total_iter_times<<" is finished"<<std::endl;
	}

	out_result_files.close();
	LOG(INFO)<<"all finished";
	return 0;
}

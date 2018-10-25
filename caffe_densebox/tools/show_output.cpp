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

using namespace caffe;
using std::vector;
using std::string;

const float THRESHOLD_DEFAULT = 0.1;

int main(int argc, char** argv) {
	if (argc < 4) {
		LOG(ERROR)<< "test_net net_proto  "
		<< "outputfolder [CPU/GPU] [device_id] [threshold]";
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
	float threshold = THRESHOLD_DEFAULT;
	if (argc > 6) {
		threshold = atof(argv[6]);
	}
	LOG(INFO) << "threshold: " << threshold;

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);

	LOG(INFO) << "Creating testing net...";
	Net<float> caffe_test_net(argv[1],caffe::TEST);
	LOG(INFO) << "Copy layers from trained net...";
	caffe_test_net.CopyTrainedLayersFrom(argv[2]);
	LOG(INFO) << "Copy Net finished...";
	FCNImageDataLayer<float>* input_data_layer;
	input_data_layer = static_cast<FCNImageDataLayer<float>*>(caffe_test_net.layers()[0].get());
	int iter_count =  input_data_layer->GetTestIterations();
	string output_folder(argv[3]);
	output_folder.append("/");
	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	for(int iter = 0; iter < iter_count; ++iter){
		Timer forward_timer;
		forward_timer.Start();
		caffe_test_net.ForwardFrom(0);
		LOG(INFO) << "Finish forwarding batch " << iter <<" in "<<
				forward_timer.MilliSeconds() << " ms, "<< "total batch_num: " << iter_count;
		const vector<Blob<float>*>& loss_layer_bottom = caffe_test_net.bottom_vecs().back();
		const vector<Blob<float>*>& data_layer_top = caffe_test_net.top_vecs().front();
		const Blob<float>* in_data = data_layer_top[0];
		const Blob<float>* labels = loss_layer_bottom[1];
		const Blob<float>* predicted = loss_layer_bottom[0];
		LOG(INFO)<<"before showDataAndLabel";
		input_data_layer->ShowDataAndPredictedLabel(output_folder,*in_data,*labels,*predicted,threshold);
	}
	return 0;
}

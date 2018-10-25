
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

using namespace caffe;
int main(int argc, char** argv) {

	 if (argc < 3) {
	    LOG(ERROR)<< "make_ref_result.bin"<<
	    			" net_model "<<
	    			" net_proto ";
	    return 0;
	 }
	string net_model_name = string(argv[1]);
	string net_proto_name = string(argv[2]);
	int iter_num = 200;
//	openblas_set_num_threads(8);
	Net<float> caffe_test_net(net_proto_name,caffe::TEST);

	std::cout<<"net initialized";
	caffe_test_net.CopyTrainedLayersFrom(net_model_name);
	caffe_test_net.Reshape();
	vector<Blob<float>*> input_blobs = caffe_test_net.input_blobs();
	FillerParameter filler_param;
	GaussianFiller<float> filler(filler_param);
	filler.Fill(input_blobs[0]);
	CHECK_EQ(input_blobs.size(),1);
	Timer timer;
	timer.Start();
	for(int i=0; i < iter_num; ++i){
		caffe_test_net.ForwardPrefilled();
	}
	double total_time = timer.MicroSeconds();
	std::cout<<"Time cost for "<<iter_num<<" iterations is "<<total_time/1000.0<<
			" MilliSeconds, and average time is: "<<total_time/iter_num/1000.0<<" MilliSeconds."
			<<std::endl;

	return 0;
}

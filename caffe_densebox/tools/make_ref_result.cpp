
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

	 if (argc < 4) {
	    LOG(ERROR)<< "make_ref_result.bin"<<
	    			" net_model "<<
	    			" net_proto "<<
	    			" input_name "<<
	    			" output_name";
	    return 0;
	 }
	string net_model_name = string(argv[1]);
	string net_proto_name = string(argv[2]);
	string input_file_name = string(argv[3]);
	string out_file_name = string(argv[4]);

	Net<float> caffe_test_net(net_proto_name,caffe::TEST);

	std::cout<<"net initialized";
	caffe_test_net.CopyTrainedLayersFrom(net_model_name);
	caffe_test_net.Reshape();
	vector<Blob<float>*> input_blobs = caffe_test_net.input_blobs();
	CHECK_EQ(input_blobs.size(),1);
	float* input_data = input_blobs[0]->mutable_cpu_data();
	FILE* in_fd = fopen(input_file_name.c_str(),"r");
	for(int i=0; i < input_blobs[0]->count(); ++i){
		fscanf(in_fd,"%f",input_data+i);
	}
	fclose(in_fd);
	vector<Blob<float>*> output_blobs = caffe_test_net.ForwardPrefilled();
	FILE* fd = fopen(out_file_name.c_str(),"w");
	const float* blob_data = output_blobs[0]->cpu_data();
	for(int h = 0; h < output_blobs[0]->height(); ++h){
		for(int w=0; w < output_blobs[0]->width(); ++w){
			for(int c = 0; c < output_blobs[0]->channels(); ++c){
				fprintf(fd, "%f ",blob_data[output_blobs[0]->offset(0,c,h,w)]);
			}
		}
	}
//	for(int i=0; i < output_blobs[0]->count(); ++i){
//		fprintf(fd, "%f ",blob_data[i]);
//	}
	fclose(fd);
	return 0;
}

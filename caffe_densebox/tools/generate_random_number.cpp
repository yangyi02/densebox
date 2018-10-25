
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
	    LOG(ERROR)<< "generate_random_number.bin"<<
	    			" output_name"<<
	    			" number ";
	    return 0;
	 }
	string out_file_name = string(argv[1]);
	FILE* fd = fopen(out_file_name.c_str(),"w");
	int number = atoi(argv[2]);
	Blob<float> blob;
	blob.Reshape(1,1,1,number);
	FillerParameter filler_param;
	GaussianFiller<float> filler(filler_param);
	filler.Fill(&blob);
	const float* blob_data = blob.cpu_data();
	for(int i=0; i < number; ++i){
		fprintf(fd, "%f ",blob_data[i]);
	}
	fclose(fd);
	return 0;
}

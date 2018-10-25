


#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/buffered_reader.hpp"

using namespace caffe;
int main(int argc, char** argv) {
	BufferedColorJPGReader<float> * buffered_reader = new BufferedColorIMGAndAVIReader<float>("",5);
	buffered_reader->Show("divx4803.avi_1222","1.jpg");

	return 0;
}

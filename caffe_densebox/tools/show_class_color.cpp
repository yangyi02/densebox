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


int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR)<< "show_class_color.bin class_list  "
		<< "output_name ";
		return 0;
	}
	vector<string> class_names_;
	string file_list_name = string(argv[1]);
	LOG(INFO) << "Opening class list file " << file_list_name;
	std::ifstream infile(file_list_name.c_str());
	CHECK(infile.good());
	string class_name;
	while(std::getline(infile,class_name)){
		if(class_name.empty()) continue;
		class_names_.push_back(class_name);
	}
	infile.close();

	string out_name = string(argv[2]);
	ShowClassColor(class_names_,out_name);
	LOG(INFO)<<"saving img : "<<out_name<<".jpg";
	return 0;
}


#include <sys/stat.h>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"


using namespace caffe;

int main(int argc, char** argv) {
	//std::cout<<"before check"<<std::endl;
  if (argc < 3) {
    LOG(ERROR)<< "ModelConvert.bin"<<
    			" Model_proto_txt  "<<
    			" trained_Model "<<
    			" dest_model_name";
    return 0;
  }

  string model_proto_txt = string(argv[1]);
  string trained_model = string(argv[2]);
  NetParameter net_param;
  ReadProtoFromTextFile(model_proto_txt, &net_param);

  LOG(INFO) << "Creating net from "<<model_proto_txt;
  Net<float> cur_net(net_param);


  LOG(INFO) << "Copying layers from trained net from :"<<trained_model;
  cur_net.CopyTrainedLayersFrom(trained_model);
  cur_net.ToProto(&net_param,false);
  string out_name = string(argv[3])+ string(".caffemodel");
  LOG(INFO) << "Writing model to :"<<out_name;
  WriteProtoToBinaryFile(net_param, out_name.c_str());

  return 0;
}

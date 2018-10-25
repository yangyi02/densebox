
#include <sys/stat.h>

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

using namespace caffe;
using std::vector;
using std::string;



void generate_response_map(Net<float>&caffe_test_net,int last_layer_id,int valid_dist ,
		float initial_value  )
{
	const vector<Blob<float>*>& last_layer_top = caffe_test_net.top_vecs()[last_layer_id];

	int c = last_layer_top[0]->channels();
	int h =  last_layer_top[0]->height();
	int w =  last_layer_top[0]->width();

	for(int i = 0; i< caffe_test_net.top_vecs().size(); ++i)
	{
			const vector<Blob<float>*>& top_data = caffe_test_net.top_vecs()[i];
			for(int j= 0 ; j < top_data.size(); ++j)
			{
				caffe::caffe_set(top_data[j]->count(),float(0),top_data[j]->mutable_cpu_diff());
			}
	}

		// set backward region
	for(int i=0; i < c ; ++i)
	{
		for(int h_diff = valid_dist*-1 ; h_diff <= valid_dist; ++h_diff  )
		{
			for(int w_diff = valid_dist*-1 ; w_diff <= valid_dist; ++w_diff  )
			{
				int new_h = int(h/2)+ h_diff;
				int new_w = int(w/2) + w_diff;
				if(new_h < 0 || new_h >= h || new_w < 0 || new_w >= w)
					continue;
				last_layer_top[0]->mutable_cpu_diff()[last_layer_top[0]->offset(0,i,new_h,new_w)] = initial_value;
			}
		}
	}

	// initial all intermediate result layer.
	for(int i = 0; i< caffe_test_net.bottom_vecs().size(); ++i)
	{
		const vector<Blob<float>*>& bottom_data = caffe_test_net.bottom_vecs()[i];
		for(int j= 0 ; j < bottom_data.size(); ++j)
		{
			caffe::caffe_set(bottom_data[j]->count(),initial_value,bottom_data[j]->mutable_cpu_data());
		}
	}

	for(int i = 0; i< caffe_test_net.top_vecs().size(); ++i)
	{
			const vector<Blob<float>*>& top_data = caffe_test_net.top_vecs()[i];
			for(int j= 0 ; j < top_data.size(); ++j)
			{
				caffe::caffe_set(top_data[j]->count(),initial_value,top_data[j]->mutable_cpu_data());
			}
	}


	// backward and calculate accumulated diff
	CHECK(caffe_test_net.bottom_vecs().size()==caffe_test_net.top_vecs().size());
	for (int i= last_layer_id; i > 0; i--)
	{
		caffe_test_net.BackwardFromTo(i,i-1);
		const	vector<Blob<float>*>& top_data = caffe_test_net.top_vecs()[i];
		LOG(INFO)<<" Dealing with layer "<<i<<" with type "<<caffe_test_net.layers()[i]->type();
		for(int j= 0 ; j < top_data.size(); ++j)
		{
			LOG(INFO)<< "square of top diff of layer "<<i<<" "<<
					caffe::caffe_cpu_dot(top_data[j]->count(),top_data[0]->mutable_cpu_diff(),top_data[0]->mutable_cpu_diff());
			LOG(INFO)<< "square of top data of layer "<<i<<" "<<
								caffe::caffe_cpu_dot(top_data[j]->count(),top_data[0]->mutable_cpu_data(),top_data[0]->mutable_cpu_data());
		}
		const	vector<Blob<float>*>& bot_data = caffe_test_net.bottom_vecs()[i];
		for(int j= 0 ; j < bot_data.size(); ++j)
		{
			LOG(INFO)<< "square of bot diff of layer "<<i<<" "<<
					caffe::caffe_cpu_dot(bot_data[j]->count(),bot_data[0]->mutable_cpu_diff(),bot_data[0]->mutable_cpu_diff());
			LOG(INFO)<< "square of bot data of layer "<<i<<" "<<
								caffe::caffe_cpu_dot(bot_data[j]->count(),bot_data[0]->mutable_cpu_data(),bot_data[0]->mutable_cpu_data());
		}
	}

	LOG(INFO)<<"backward finished.";
}


int main(int argc, char** argv) {
	if (argc < 4) {
		LOG(ERROR)<< "net_proto  "
		<< "outputfolder [CPU/GPU] [device_id]  ";
		return 0;
	}

//	Caffe::set_phase(Caffe::TRAIN);

	if (argc > 3 && strcmp(argv[3], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);

	} else {
	Caffe::set_mode(Caffe::GPU);
	int device_id = 0;
	if (argc > 4) {
	  device_id = atoi(argv[4]);
	}
	Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
	}

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	NetParameter response_net_param;
	int first_layer_id, last_layer_id;
	Net<float>::ToResponseNet(test_net_param,&response_net_param,&first_layer_id,&last_layer_id);

	int ave_times = 1;


	LOG(INFO) << "Creating testing net...";
	Net<float> caffe_test_net(response_net_param);
	LOG(INFO)<<"first layer id: "<<first_layer_id <<"   last_layer_id: "<<last_layer_id;
	caffe_test_net.ForwardPrefilled();
	LOG(INFO) << "Forward finished...";
	string output_folder(argv[2]);
	output_folder.append("/");
	mkdir(output_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	vector<int> params_jpg;
	params_jpg.push_back(CV_IMWRITE_JPEG_QUALITY);
	params_jpg.push_back(100);

	vector<int> params_png;
	params_png.push_back(CV_IMWRITE_PNG_COMPRESSION);
	params_png.push_back(9);
	float initial_value = 1;
	int valid_dist = 0;


	for(int end_layer = last_layer_id; end_layer >first_layer_id; end_layer -- )
	{
		const vector<Blob<float>*>& bottom_data = caffe_test_net.bottom_vecs()[first_layer_id];
		int height = bottom_data[0]->height();
		int width = bottom_data[0]->width();
		int channels = bottom_data[0]->channels();
		LOG(INFO)<<"c: "<<channels<<" height: "<<height<<" width: "<<width;
		Blob<float> cur_res;
		cur_res.ReshapeLike(*(bottom_data[0]));
		caffe::caffe_set(cur_res.count(),float(0),cur_res.mutable_cpu_diff());
		for(int i=0; i < ave_times; ++i){
			generate_response_map( caffe_test_net,  end_layer, valid_dist, initial_value);
			float max = FLT_MIN;
			float min = FLT_MAX;
			// get min and max
			for (int c = 0; c < channels ; ++c) {
				for (int h = 0; h <  height ; ++h) {
				  for (int w = 0; w <  width ; ++w) {
					max = MAX(bottom_data[0]->diff_at(0, c, h, w),max) ;
					min = MIN(bottom_data[0]->diff_at(0, c, h, w),max);
				  }
				}
			}

		//    save one channel of input
			  cv::Mat cv_img_original( height ,  width , CV_8UC3);
			  for (int c = 0; c < channels ; ++c) {
				for (int h = 0; h <  height ; ++h) {
				  for (int w = 0; w <  width ; ++w) {
					  cur_res.mutable_cpu_diff()[cur_res.offset(0,c,h,w)] +=
						255 * MIN(1, MAX(0, (bottom_data[0]->diff_at(0, c, h, w) - min)/(max-min) ));
				  }
				}
			  }
//
//			caffe::caffe_add(cur_res.count(),cur_res.cpu_diff(),
//					bottom_data[0]->cpu_diff(), cur_res.mutable_cpu_diff());
		}

		float max = FLT_MIN;
		float min = FLT_MAX;

		// get min and max
		for (int c = 0; c < channels ; ++c) {
			for (int h = 0; h <  height ; ++h) {
			  for (int w = 0; w <  width ; ++w) {
				max = MAX(cur_res.diff_at(0, c, h, w),max) ;
				min = MIN(cur_res.diff_at(0, c, h, w),max);
			  }
			}
		}

	//    save one channel of input
		  cv::Mat cv_img_original( height ,  width , CV_8UC3);
		  for (int c = 0; c < channels ; ++c) {
			for (int h = 0; h <  height ; ++h) {
			  for (int w = 0; w <  width ; ++w) {
				cv_img_original.at<cv::Vec3b>(h, w)[c]
				= 255 * MIN(1, MAX(0, (cur_res.diff_at(0, c, h, w) - min)/(max-min) )) ;
			  }
			}
		  }

		  char path[256];

		  sprintf(path, "%s/response_map_layer_%d_%s.txt", output_folder.c_str(),
				  end_layer, caffe_test_net.layer_names()[end_layer].c_str());
		  std::ofstream out_pred(path);

		  for (int h = 0; h <  height ; h++) {
			  for (int w = 0; w < width ; w++) {
				out_pred << cur_res.diff_at(0, 0, h, w) << " ";
			  }
			  out_pred << "\n";
		  }
		  out_pred.close();

		  sprintf(path, "%s/response_map_layer_%d_%s.jpg", output_folder.c_str(),
				  end_layer,caffe_test_net.layer_names()[end_layer].c_str());
		  imwrite(path, cv_img_original, params_jpg);
		  LOG(INFO)<<"layer "<< end_layer<<" is done.";
	}


	  LOG(INFO)<<"ALL Done.";
	  return 0;
}

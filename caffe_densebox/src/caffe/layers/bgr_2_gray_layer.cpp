#include <vector>

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BGR2GrayLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->channels(),3);
	bgr_weight_.Reshape(1,1,1,3);
	Dtype* bgr_weight = bgr_weight_.mutable_cpu_data();
	bgr_weight[0] = 0.114;
	bgr_weight[1] = 0.587;
	bgr_weight[2] = 0.299;
}

template <typename Dtype>
void BGR2GrayLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	int num = bottom[0]->num();
	int channels = 1;
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	top[0]->Reshape(num, channels, height, width);
}


//Gray = R*0.299 + G*0.587 + B*0.114
template <typename Dtype>
void BGR2GrayLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Dtype*  bgr_weight = bgr_weight_.mutable_cpu_data();
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int size_per_channel =  height * width;
	int spacial_size = bottom[0]->count()/num;

	for(int i = 0; i < num; ++i){
		Dtype* in_data = bottom[0]->mutable_cpu_data() + spacial_size * i;
		Dtype* out_data = top[0]->mutable_cpu_data() + size_per_channel * i;
		caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,size_per_channel,1,3,
				1.,in_data,bgr_weight,0.,out_data);
	}
}


#ifdef CPU_ONLY
STUB_GPU(BGR2GrayLayer);
#endif

INSTANTIATE_CLASS(BGR2GrayLayer);
REGISTER_LAYER_CLASS(BGR2Gray);

}  // namespace caffe

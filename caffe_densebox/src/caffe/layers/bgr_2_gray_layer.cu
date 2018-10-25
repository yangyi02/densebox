#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {


template <typename Dtype>
void BGR2GrayLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype*  bgr_weight = bgr_weight_.mutable_gpu_data();
	int num = bottom[0]->num();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int size_per_channel =  height * width;
	int spacial_size = bottom[0]->count()/num;
	for(int i = 0; i < num; ++i){
		Dtype* in_data = bottom[0]->mutable_gpu_data() + spacial_size * i;
		Dtype* out_data = top[0]->mutable_gpu_data() + size_per_channel * i;
		caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,size_per_channel,1,3,
				1.,in_data,bgr_weight,0.,out_data);
	}
}
INSTANTIATE_LAYER_GPU_FUNCS(BGR2GrayLayer);


}  // namespace caffe

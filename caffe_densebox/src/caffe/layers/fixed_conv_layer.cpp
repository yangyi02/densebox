#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/fixed_conv_layer.hpp"
#include "caffe/blob_transform.hpp"
namespace caffe {

template <typename Dtype>
void FixedConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom,top);
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  this->input_bound_scale_ = conv_param.input_bound_scale();
  this->weight_bound_scale_ = conv_param.weight_bound_scale();
  CHECK_GT(this->input_bound_scale_ , 0);
  CHECK_GT(this->weight_bound_scale_ , 0);
  this->weight_backup_.Reshape(this->blobs_[0]->shape());
}

template <typename Dtype>
void FixedConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	// fix weight by max(abs(min), abs(max)) and bound
	Blob<Dtype>& conv_weight = *(this->blobs_[0].get());
	caffe::caffe_copy(conv_weight.count(), conv_weight.cpu_data(),
			this->weight_backup_.mutable_cpu_data());
	Dtype weight_bound_scale = this->weight_bound_scale_ -1;
	fxnet_transform<op::BoundByMaxAbs>(conv_weight,conv_weight,weight_bound_scale);

	// fix input by min , max and bound
	Dtype input_bound_scale = this->input_bound_scale_-1;
	for(int i=0; i < bottom.size(); ++i){
		Blob<Dtype>& input_blob = *(bottom[i]);
		fxnet_transform<op::BoundByMinMax>(input_blob,input_blob, input_bound_scale);
	}

	ConvolutionLayer<Dtype>::Forward_cpu(bottom, top);
}

template <typename Dtype>
void FixedConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	// restore weight
	Blob<Dtype>& conv_weight = *(this->blobs_[0].get());
	caffe::caffe_copy(conv_weight.count(), this->weight_backup_.cpu_data(),
			conv_weight.mutable_cpu_data());

	ConvolutionLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(FixedConvolutionLayer);
#endif

INSTANTIATE_CLASS(FixedConvolutionLayer);
REGISTER_LAYER_CLASS(FixedConvolution);
}  // namespace caffe

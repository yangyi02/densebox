#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/fcn_data_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LabelRelatedDropoutForward(const int n, const Dtype* in,
    const  int* mask,   Dtype* out, const Dtype value_masked_) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * mask[index] +(mask[index]==0)*value_masked_ ;
  }
}

template <typename Dtype>
__global__ void LabelRelatedDropoutSetPosMask(const int n, const Dtype* label_data,
		int* mask_data){
	CUDA_KERNEL_LOOP(index, n) {
		Dtype label_value =  (label_data[index]);
		mask_data[index] = label_value == Dtype(0) ? 0:1 ;
	}
}

template<typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::set_mask_for_positive_gpu(const vector<Blob<Dtype>*>& bottom){
	const int count = bottom[1]->count();
	const Dtype * label_data = bottom[1]->gpu_data();
	int* mask_data = this->mask_vec_.mutable_gpu_data();
	LabelRelatedDropoutSetPosMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	        count, label_data,mask_data);
	CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	if(this->negative_ratio_ <= 0)
		set_mask_for_positive_gpu(bottom);
	else
		this->set_mask_from_labels_cpu_parallel(bottom);
	if(this->pic_print_)
		PrintMask();
	const int* mask =static_cast<const int*>(this->mask_vec_.gpu_data());
	// set thresholds
	// NOLINT_NEXT_LINE(whitespace/operators)
	LabelRelatedDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, mask, top_data,value_masked_);
	CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void LabelRelatedDropoutBackward(const int n, const Dtype* in_diff,
    const  int* mask, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] *mask[index];
  }
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int* mask =
	  static_cast<const  int*>(this->mask_vec_.gpu_data());
	const int count = bottom[0]->count();
	// NOLINT_NEXT_LINE(whitespace/operators)
	LabelRelatedDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
	CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, mask,bottom_diff);
	CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LabelRelatedDropoutLayer);


}  // namespace caffe

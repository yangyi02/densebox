#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void SmoothL2LossKernel(const int count, const Dtype* src, Dtype* dst){
	CUDA_KERNEL_LOOP(index, count) {
		Dtype in_data = src[index];
		if( in_data > 1){
			dst[index] = in_data - 0.5;
		}else if(in_data < -1){
			dst[index] = - in_data - 0.5;
		}else{
			dst[index] = 0.5 * in_data * in_data;
		}
	}
}

template <typename Dtype>
__global__ void  ThresholdDiffKernel(const int count, Dtype* dst, Dtype* weight,
		const Dtype* label, Dtype thred){
	CUDA_KERNEL_LOOP(index, count) {
		Dtype in_data = dst[index];
		Dtype label_data = label[index];
		Dtype cur_weight = (std::abs(in_data) > thred && std::abs(in_data - label_data)/label_data > 0.3);
		weight[index] = cur_weight;
		dst[index] = cur_weight * in_data;
	}
}


template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype penalize_threshold = this->layer_param_.loss_param().penalize_threshold();

	bool smooth = this->layer_param_.loss_param().smooth();
	bool need_normalization_per_positive = (this->layer_param_.has_loss_param() &&
		  this->layer_param_.loss_param().normalize_per_positive());
	bool need_normalize = (this->layer_param_.has_loss_param() &&
			  this->layer_param_.loss_param().normalize());
	int count = bottom[0]->count();
	caffe_gpu_sub(
		count,
		bottom[0]->gpu_data(),
		bottom[1]->gpu_data(),
		diff_.mutable_gpu_data());

	Dtype activated_sample_num = 0;

  if(penalize_threshold > 0){
  	ThresholdDiffKernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  			(count, diff_.mutable_gpu_data(),one_.mutable_gpu_data(),
  					bottom[1]->gpu_data(),penalize_threshold);
  	caffe_gpu_dot(count, one_.gpu_data(), one_.gpu_data(),&activated_sample_num);
  	LOG(INFO)<<"activated_sample_num ratio: "<< activated_sample_num / count;
  }

	Dtype dot;
	Dtype scale = this->layer_param_.loss_param().scale();
	caffe::caffe_gpu_scal(count,scale,diff_.mutable_gpu_data());
	Dtype loss = 0;



	caffe::caffe_gpu_set(count,Dtype(1),one_.mutable_gpu_data());
	if(smooth == false){
		caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
		loss = dot / bottom[0]->num() / Dtype(2);
	}
	else{
		SmoothL2LossKernel<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				 count, diff_.gpu_data(),diff_.mutable_gpu_diff());
		CUDA_POST_KERNEL_CHECK;
		caffe_gpu_dot(count, diff_.gpu_diff(), one_.gpu_data(), &dot);
		loss = dot / bottom[0]->num() ;
	}

	this->scale_factor_  = 1;
	if(need_normalize){
		this->scale_factor_ =  bottom[0]->count()/bottom[0]->num();
		loss /= this->scale_factor_ ;
	}
	else if(need_normalization_per_positive)
	{
	  const int gt_bottom_id =  this->layer_param_.loss_param().label_bottom_id();
	  CHECK(gt_bottom_id < bottom.size());
	  caffe::caffe_gpu_sign(count,bottom[gt_bottom_id]->gpu_data(),one_.mutable_gpu_diff());
	  caffe::caffe_gpu_abs(count,one_.gpu_diff(),one_.mutable_gpu_diff());
	  Dtype sum;
	  caffe_gpu_dot(count,
			  one_.gpu_diff(), one_.gpu_data(),&sum);
	  this->scale_factor_ = ((sum+1))/ bottom[0]->num();
	  loss /= this->scale_factor_;
	}

	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  bool smooth = this->layer_param_.loss_param().smooth();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num()/this->scale_factor_;
      if(smooth ){
		  caffe::caffe_gpu_scalar_max(bottom[i]->count(),Dtype(-1),diff_.gpu_data(),diff_.mutable_gpu_data());
		  caffe::caffe_gpu_scalar_min(bottom[i]->count(),Dtype(1),diff_.gpu_data(),diff_.mutable_gpu_data());
	  }
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe

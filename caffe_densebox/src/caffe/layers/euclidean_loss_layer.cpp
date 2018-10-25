#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL2Loss_cpu(const int count,  const Dtype* src,Dtype* dst){
	for(int i=0 ; i < count; ++i){
		Dtype in_data = src[i];
		if( in_data > 1){
			dst[i] = in_data - 0.5;
		}else if(in_data < -1){
			dst[i] = - in_data - 0.5;
		}else{
			dst[i] = 0.5 * in_data * in_data;
		}
	}
}

template <typename Dtype>
void threshold_diff_cpu(const int count, Dtype* dst, Dtype* weight, const Dtype* label, Dtype thred){
	for(int i=0 ; i < count; ++i){
		Dtype in_data = dst[i];
		Dtype label_data = label[i];
		Dtype cur_weight = (std::abs(in_data) > thred && std::abs(in_data - label_data)/label_data > 0.3);
		weight[i] = cur_weight;
		dst[i] = cur_weight * in_data;
	}
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  one_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype penalize_threshold = this->layer_param_.loss_param().penalize_threshold();

	bool smooth = this->layer_param_.loss_param().smooth();
	bool need_normalization_per_positive = (this->layer_param_.has_loss_param() &&
			this->layer_param_.loss_param().normalize_per_positive());
	bool need_normalize = (this->layer_param_.has_loss_param() &&
			this->layer_param_.loss_param().normalize());

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  Dtype activated_sample_num = 0;
  if(penalize_threshold > 0){
  	threshold_diff_cpu(count, diff_.mutable_cpu_data(),one_.mutable_cpu_data(),
  			bottom[1]->cpu_data(),penalize_threshold);
  	activated_sample_num = caffe_cpu_dot(count, one_.cpu_data(), one_.cpu_data());
  	LOG(INFO)<<"activated_sample_num ratio: "<< activated_sample_num / count;
  }


	Dtype scale = this->layer_param_.loss_param().scale();
	caffe::caffe_scal(count,scale,diff_.mutable_cpu_data());
	Dtype loss = 0;


	caffe::caffe_set(count,Dtype(1),one_.mutable_cpu_data());
	if(smooth == false){
		Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
		loss = dot / bottom[0]->num() / Dtype(2);
	}else{
		SmoothL2Loss_cpu(count, diff_.cpu_data(),diff_.mutable_cpu_diff() );
		Dtype dot = caffe_cpu_dot(count, one_.cpu_data(), diff_.cpu_diff());
		loss = dot / bottom[0]->num() ;
	}


	this->scale_factor_  = 1;

	if(need_normalize){
		this->scale_factor_ =  bottom[0]->count()/bottom[0]->num();

		loss /= this->scale_factor_ ;

	}
	else if(need_normalization_per_positive){
		const int gt_bottom_id =  this->layer_param_.loss_param().label_bottom_id();
		CHECK(gt_bottom_id < bottom.size());
		caffe::caffe_cpu_sign(count,bottom[gt_bottom_id]->cpu_data(),one_.mutable_cpu_diff());
		caffe::caffe_abs(count,one_.cpu_diff(),one_.mutable_cpu_diff());
		Dtype sum = caffe_cpu_dot(count,
			  one_.cpu_diff(), one_.cpu_data()) +1;
		this->scale_factor_ =  sum / bottom[0]->num();
		loss /= this->scale_factor_ ;
	}

	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool smooth = this->layer_param_.loss_param().smooth();

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num()/this->scale_factor_;

      if(smooth ){
    	  caffe::caffe_scalar_max(bottom[i]->count(),Dtype(-1),diff_.cpu_data(),diff_.mutable_cpu_data());
    	  caffe::caffe_scalar_min(bottom[i]->count(),Dtype(1),diff_.cpu_data(),diff_.mutable_cpu_data());
      }
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe

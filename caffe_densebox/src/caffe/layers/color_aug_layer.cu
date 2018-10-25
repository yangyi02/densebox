#include <vector>

#include "caffe/layers/color_aug_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
namespace caffe {


template <typename Dtype>
__global__ void color_aug_kernel(const int data_dim, const Dtype* src, Dtype* dst, const int batch_num,
		const int spatial_dim, const Dtype* batch_bgr_perturb_scale, const Dtype* batch_intensity_scale,
		Dtype mean_b, Dtype mean_g, Dtype mean_r){
	Dtype mean_bgr[3];
	mean_bgr[0] = mean_b;
	mean_bgr[1] = mean_g;
	mean_bgr[2] = mean_r;
	CUDA_KERNEL_LOOP(index, data_dim) {
		int batch_id = index/( spatial_dim*3);
		int c_id = (index/spatial_dim)%3;
		Dtype value = (src[index] + mean_bgr[c_id]) * batch_bgr_perturb_scale[c_id +
										batch_id*3] *batch_intensity_scale[batch_id] - mean_bgr[c_id];
		dst[index] = value;
	}
}

template <typename Dtype>
__global__ void color_aug_to_gray_kernel(const int loop_dim, const Dtype* src, Dtype* dst, const int gray_batch_id_size,
		const int spatial_dim, const int* gray_batch_ids, Dtype b_weight, Dtype g_weight, Dtype r_weight){


	CUDA_KERNEL_LOOP(index, loop_dim) {
		int batch_id_id = index/( spatial_dim);
		int batch_id = gray_batch_ids[batch_id_id];
		int pixel_id = index%(spatial_dim);
		int src_idx =  batch_id * batch_id * 3 + pixel_id;
		Dtype value =  src[src_idx]*b_weight + src[src_idx + spatial_dim]*g_weight + src[src_idx + spatial_dim*2]*r_weight;
		dst[src_idx] = value;
		dst[src_idx + spatial_dim] = value;
		dst[src_idx + spatial_dim*2] = value;
	}
}


template <typename Dtype>
void ColorAugLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	bgr_weight_[0] = 0.114;
	bgr_weight_[1] = 0.587;
	bgr_weight_[2] = 0.299;
  for(int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id){

  	int batch_n = bottom[bottom_id]->num();
  	this->blob_bgr_perturb_scale_.Reshape(1,1,1,batch_n*3);
  	this->blob_intensity_scale_.Reshape(1,1,1,batch_n);
  	this->blob_gray_flag_.Reshape(1,1,1,batch_n);
  	gray_ids_.clear();
  	Dtype* batch_bgr_perturb_scale_ = blob_bgr_perturb_scale_.mutable_cpu_data();
  	Dtype* batch_intensity_scale_ =  blob_intensity_scale_.mutable_cpu_data();
  	Dtype* should_be_gray_ = blob_gray_flag_.mutable_cpu_data();

  	for(int i=0; i < batch_n; ++i){
  		batch_bgr_perturb_scale_[0 + i*3] = 1 + (this->RandFloat()* 2 - 1)* channel_perturb_range_ ;
  		batch_bgr_perturb_scale_[1 + i*3] = 1 + (this->RandFloat()* 2 - 1)* channel_perturb_range_ ;
  		batch_bgr_perturb_scale_[2 + i*3] = 3 - batch_bgr_perturb_scale_[1 + i*3]- batch_bgr_perturb_scale_[0 + i*3];
  		batch_intensity_scale_[i] = 1 + (this->RandFloat()* 2 - 1)* this->intensity_perturb_range_ ;
		//LOG(INFO)<<"batch_bgr_perturb_scale_ i : "<<batch_bgr_perturb_scale_[0 + i*3]<<" "<<batch_bgr_perturb_scale_[1 + i*3]<<" "
		//	<<batch_bgr_perturb_scale_[2 + i*3]<<" batch_intensity_scale_[i]: "<<batch_intensity_scale_[i];
  		if(this->RandFloat() < this->gray_ratio_) {
  			should_be_gray_[i] = 1;
  			gray_ids_.push_back(i);
  		}else{
  			should_be_gray_[i] = 0;
  		}
  	}
  	this->blob_gray_ids_.Reshape(1,1,1,gray_ids_.size()+1);
	if(gray_ids_.size() > 0)
	  	memcpy(blob_gray_ids_.mutable_cpu_data(),&(gray_ids_[0]),gray_ids_.size());
//	LOG(INFO)<<"gray_ids_.size(): "<<gray_ids_.size()<<" batch_n: "<<batch_n;
    const int count = top[bottom_id]->count();
    const int spatial_dim = bottom[bottom_id]->offset(0, 1,0,0);
  	color_aug_kernel<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>(
  			count,bottom[bottom_id]->gpu_data(), top[bottom_id]->mutable_gpu_data(),batch_n,
  			spatial_dim,blob_bgr_perturb_scale_.gpu_data(),blob_intensity_scale_.gpu_data(),mean_bgr_[0],
  			mean_bgr_[1],mean_bgr_[2]);
//	CUDA_POST_KERNEL_CHECK;
	if(gray_ids_.size() > 0){
	  	int loop_n = spatial_dim*gray_ids_.size();
  		color_aug_to_gray_kernel<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS>>>(
  			loop_n,top[bottom_id]->gpu_data(),top[bottom_id]->mutable_gpu_data(),gray_ids_.size(),
  			spatial_dim, blob_gray_ids_.gpu_data(),bgr_weight_[0],bgr_weight_[1],bgr_weight_[2]);
//		CUDA_POST_KERNEL_CHECK;
	}
  }
//LOG(INFO)<<"finish forward"<<std::endl;
}

INSTANTIATE_LAYER_GPU_FUNCS(ColorAugLayer);

}  // namespace caffe

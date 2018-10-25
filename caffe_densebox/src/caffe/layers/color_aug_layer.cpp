#include <vector>

#include "caffe/layers/color_aug_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"
#include <stdio.h>
namespace caffe {

template <typename Dtype>
void ColorAugLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
//  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
//    "allow in-place computation.";
  for(int i=0; i < bottom.size(); ++i){
  	CHECK_EQ(bottom[i]->channels(),3)<<"The channel number of bottom "<<i
  			<<" should be 3. "<<std::endl;
  }
  const ColorAugParameter& color_aug_param = this->layer_param_.color_aug_param();
  this->gray_ratio_ = color_aug_param.gray_ratio();
  CHECK_LE(this->gray_ratio_,1);
  CHECK_GE(this->gray_ratio_,0);
  this->channel_perturb_range_ = color_aug_param.channel_perturb_range();
  CHECK_LE(this->channel_perturb_range_,1);
	CHECK_GE(this->channel_perturb_range_,0);
  this->intensity_perturb_range_ = color_aug_param.intensity_perturb_range();
  CHECK_LE(this->intensity_perturb_range_,1);
	CHECK_GE(this->intensity_perturb_range_,0);

	this->mean_bgr_[0] = color_aug_param.mean_b();
	this->mean_bgr_[1] = color_aug_param.mean_g();
	this->mean_bgr_[2] = color_aug_param.mean_r();

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));


}


template <typename Dtype>
void ColorAugLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	bgr_weight_[0] = 0.114;
	bgr_weight_[1] = 0.587;
	bgr_weight_[2] = 0.299;
  for(int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id){
  	int batch_n = bottom[bottom_id]->num();
  	this->blob_bgr_perturb_scale_.Reshape(1,1,1,batch_n*3);
  	this->blob_intensity_scale_.Reshape(1,1,1,batch_n);
  	this->blob_gray_flag_.Reshape(1,1,1,batch_n);

  	Dtype* batch_bgr_perturb_scale_ = blob_bgr_perturb_scale_.mutable_cpu_data();
  	Dtype* batch_intensity_scale_ =  blob_intensity_scale_.mutable_cpu_data();
  	Dtype* should_be_gray_ = blob_gray_flag_.mutable_cpu_data();


  	for(int i=0; i < batch_n; ++i){
  		batch_bgr_perturb_scale_[0 + i*3] = 1 + (this->RandFloat()* 2 - 1)* channel_perturb_range_ ;
  		batch_bgr_perturb_scale_[1 + i*3] = 1 + (this->RandFloat()* 2 - 1)* channel_perturb_range_ ;
  		batch_bgr_perturb_scale_[2 + i*3] = 3 - batch_bgr_perturb_scale_[1 + i*3]- batch_bgr_perturb_scale_[0 + i*3];
  		batch_intensity_scale_[i] = 1 + (this->RandFloat()* 2 - 1)* this->intensity_perturb_range_ ;
  		should_be_gray_[i] = (this->RandFloat() < this->gray_ratio_)? 1 : 0;
  	}

//    const int count = top[bottom_id]->count();
    Dtype* top_data = top[bottom_id]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[bottom_id]->cpu_data();
    const int spatial_dim = bottom[bottom_id]->offset(0, 1,0,0);

    for(int n_id = 0; n_id < bottom[bottom_id]->num(); ++n_id){
			for(int c_id = 0; c_id < bottom[bottom_id]->channels(); ++c_id){
				const Dtype* cur_bottom_data = bottom_data + bottom[bottom_id]->offset(n_id, c_id,0,0);
				Dtype* cur_top_data = top_data + top[bottom_id]->offset(n_id, c_id,0,0);

				caffe::caffe_copy(spatial_dim,cur_bottom_data,cur_top_data);
				caffe::caffe_add_scalar(spatial_dim, this->mean_bgr_[c_id],cur_top_data);
				caffe::caffe_cpu_scale(spatial_dim, batch_bgr_perturb_scale_[c_id + n_id*3] *batch_intensity_scale_[n_id],
						cur_top_data, cur_top_data);
				caffe::caffe_add_scalar(spatial_dim, -1*this->mean_bgr_[c_id],cur_top_data);
			}
			// to gray image if need

			if(should_be_gray_[n_id] != Dtype(0)){
				for(int c_id = 0; c_id < bottom[bottom_id]->channels(); ++c_id){
					const Dtype* cur_bottom_data = bottom_data + bottom[bottom_id]->offset(n_id, c_id,0,0);
					Dtype* cur_top_data = top_data + top[bottom_id]->offset(n_id, c_id,0,0);
					caffe::caffe_cpu_scale(spatial_dim,bgr_weight_[c_id] ,
							cur_bottom_data, cur_top_data);
					if(c_id > 0){
						Dtype* cur_top_data_c0 = top_data + top[bottom_id]->offset(n_id, 0,0,0);
						caffe::caffe_add(spatial_dim,cur_top_data_c0,cur_top_data,cur_top_data_c0);
					}
				}

				for(int c_id = 1; c_id < bottom[bottom_id]->channels(); ++c_id){
					Dtype* cur_top_data_c0 = top_data + top[bottom_id]->offset(n_id, 0,0,0);
					Dtype* cur_top_data = top_data + top[bottom_id]->offset(n_id, c_id,0,0);
					caffe::caffe_copy(spatial_dim,cur_top_data_c0,cur_top_data);
				}
			}


    }
  }

}



#ifdef CPU_ONLY
STUB_GPU(ColorAugLayer);
#endif

INSTANTIATE_CLASS(ColorAugLayer);
REGISTER_LAYER_CLASS(ColorAug);

}  // namespace caffe

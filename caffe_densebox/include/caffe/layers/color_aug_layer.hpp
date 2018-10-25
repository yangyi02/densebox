#ifndef CAFFE_COLOR_AUG_LAYER_HPP_
#define CAFFE_COLOR_AUG_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {


template <typename Dtype>
class ColorAugLayer : public NeuronLayer<Dtype> {
 public:
  explicit ColorAugLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ColorAugLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
	inline Dtype RandFloat(){
		caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		return  (*prefetch_rng)() / static_cast<Dtype>(prefetch_rng->max());
	}


  shared_ptr<Caffe::RNG> prefetch_rng_;
  Dtype gray_ratio_, channel_perturb_range_,intensity_perturb_range_;
  Dtype  bgr_weight_[3];

  Dtype  mean_bgr_[3];

  vector<int> gray_ids_;
  vector<bool>  should_be_gray_;

  Blob<Dtype> blob_bgr_perturb_scale_, blob_intensity_scale_, blob_gray_flag_;
  Blob<int> blob_gray_ids_;
};


}  // namespace caffe

#endif  // CAFFE_COLOR_AUG_LAYER_HPP_

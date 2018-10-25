#ifndef CAFFE_ELEMENT_HINGE_LOSS_LAYER_HPP_
#define CAFFE_ELEMENT_HINGE_LOSS_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Similar as HingeLossLayer. But it compute the element-wise hinge loss.
 *         Important:  bottom[0] is the the predicted result, and bottom[1] is the ground truth
 */
template <typename Dtype>
class ElementHingeLossLayer : public LossLayer<Dtype> {
 public:
  explicit ElementHingeLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), sign_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
	  return "ElementHingeLoss";
  }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> sign_;
  Blob<Dtype> one_;
  Dtype scale_factor_;

};



}  // namespace caffe

#endif  // CAFFE_ELEMENT_HINGE_LOSS_LAYER_HPP_

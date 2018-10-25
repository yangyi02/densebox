#include <functional>
#include <utility>
#include <vector>
#include <float.h>
#include "caffe/layers/label_convert_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LabelConvertLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	num_class_ = bottom[0]->shape(1);
}

template <typename Dtype>
void LabelConvertLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> bottom_shape = bottom[0]->shape();
	bottom_shape[1]  = 1;
	top[0]->Reshape(bottom_shape);
	spatial_dim_ = bottom[0]->count(2);
}

template <typename Dtype>
void LabelConvertLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	int num = bottom[0]->shape(0);
	int channel = bottom[0]->shape(1);
	for(int n = 0; n < num; ++n){
		for(int spatial_id = 0; spatial_id < spatial_dim_; ++spatial_id){
			Dtype out_label = -1;
			Dtype max_value = -FLT_MAX;
			Dtype acc_label = 0;
			Dtype acc_weight = 0;
			for(int c=0; c < channel; ++c){
				Dtype cur_weight = std::max(Dtype(0),bottom_data[(n * channel + c ) * spatial_dim_ + spatial_id ]);
				acc_label += c * cur_weight;
				acc_weight += cur_weight;
				if(bottom_data[(n * channel + c ) * spatial_dim_ + spatial_id ] > max_value){
					out_label = c;
					max_value = bottom_data[(n * channel + c ) * spatial_dim_ + spatial_id ];

//					if(out_label == Dtype(-1)){
//						out_label = c;
//					}else{
//						LOG(INFO)<<"Error, duplicated label!  channel "<< c <<" and channel "<<out_label
//								<<"in spatial_dim "<< spatial_id <<" has the same active flag.";
//					}
				}
			}
//			if(out_label < 0){
//				LOG(INFO)<<"Error:  No label is found in spatial_dim "<< spatial_id;
//			}
			top_data[n * spatial_dim_ + spatial_id] = out_label;
//			top_data[n * spatial_dim_ + spatial_id] = acc_label/acc_weight;
		}
	}


}

INSTANTIATE_CLASS(LabelConvertLayer);
REGISTER_LAYER_CLASS(LabelConvert);

}  // namespace caffe

#include <string>
#include <vector>
#include <cfloat>
#include <algorithm>
#include "caffe/layers/pyramid_data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

namespace caffe {


template <typename Dtype>
void ROI2HeatMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	const ROI2HeatMapParam& roi2heatmap_param = this->layer_param_.roi_2_heatmap_param();
	num_class_ = roi2heatmap_param.num_class();
	CHECK_GE(num_class_,1);
	CHECK(roi2heatmap_param.has_map_h());
	CHECK(roi2heatmap_param.has_map_w());
	CHECK(roi2heatmap_param.has_map_num());
	map_h_ = roi2heatmap_param.map_h();
	map_w_ = roi2heatmap_param.map_w();
	map_num_ = roi2heatmap_param.map_num();
	label_type_ = roi2heatmap_param.label_type();
	CHECK_GE(map_h_,1);
	CHECK_GE(map_w_,1);
	CHECK_GE(map_num_,1);
	top[0]->Reshape(map_num_, 5, map_h_, map_w_);
}


template <typename Dtype>
void ROI2HeatMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->num(),bottom[1]->num());
	CHECK_EQ(bottom[0]->num(),bottom[2]->num());
	CHECK_EQ(bottom[0]->num(),bottom[3]->num());
//	LOG(INFO)<<"fuck1";
	top[0]->Reshape(map_num_, num_class_*5, map_h_, map_w_);
//	LOG(INFO)<<"fuck2";
	ROI_data_length_ = bottom[0]->width();
	ROI_info_length_ = bottom[1]->width();
	CHECK_EQ(bottom[0]->width(),5);
	CHECK_EQ(bottom[1]->width(),4);
	CHECK_EQ(bottom[3]->width(),4*num_class_);
	if(label_type_ == ROI2HeatMapParam_LabelType_NPlus1)
		CHECK_EQ(bottom[2]->width(),num_class_ + 1);
	else if(label_type_ == ROI2HeatMapParam_LabelType_OneDim)
		CHECK_EQ(bottom[2]->width(), 1);
	else if(label_type_ == ROI2HeatMapParam_LabelType_NDim)
		CHECK_EQ(bottom[2]->width(), num_class_);
	else
		LOG(FATAL) << "Unknown type " << label_type_;
//	LOG(INFO)<<"fuck3";
}

/**
 * @brief Convert ROI output into HeatMap
 *		  The input has 4 blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the predicted labels and  bbox_diff.
 *		  Predicted label is a class_num +1 array, with the first channel indicating background probability.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
void ROI2HeatMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	// initialize output
	Dtype* out_data = top[0]->mutable_cpu_data();
	const int out_data_count = top[0]->count();

	caffe::caffe_set(out_data_count, Dtype(0),out_data);
	caffe::caffe_set(map_h_ * map_w_ * num_class_,Dtype(FLT_MIN), out_data);

	// convert ROI
	const int ROI_num = bottom[0]->num();
	const Dtype* ROI_data  = bottom[0]->cpu_data();
	const Dtype* ROI_info  = bottom[1]->cpu_data();
	const Dtype* ROI_predicted = bottom[2]->cpu_data();
	const Dtype* ROI_diff  = bottom[3]->cpu_data();
	int num_id, w, h;

	if(label_type_ == ROI2HeatMapParam_LabelType_NPlus1){
		for(int i=0; i < ROI_num; ++i){
			num_id =  ROI_data[i * ROI_data_length_];
			w = ROI_info[i*ROI_info_length_ +1];
			h = ROI_info[i*ROI_info_length_ +2];
			const Dtype* cur_predicted  = ROI_predicted + i*(num_class_ + 1) + 1;
			for(int class_id = 0; class_id < num_class_; ++class_id){
				out_data[top[0]->offset(num_id,class_id ,h, w)] = cur_predicted[class_id];
				out_data[top[0]->offset(num_id,num_class_+class_id*4+0,h, w)] = ROI_diff[i*4*num_class_ + class_id +0] + w - ROI_data[i * ROI_data_length_ +1 ] ;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+1,h, w)] = ROI_diff[i*4*num_class_ + class_id +1] + h - ROI_data[i * ROI_data_length_ +2 ];
				out_data[top[0]->offset(num_id,num_class_+class_id*4+2,h, w)] = ROI_diff[i*4*num_class_ + class_id +2] + w - ROI_data[i * ROI_data_length_ +3 ];;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+3,h, w)] = ROI_diff[i*4*num_class_ + class_id +3] + h - ROI_data[i * ROI_data_length_ +4 ];;
			}
		}
	}else if(label_type_ == ROI2HeatMapParam_LabelType_OneDim){
		for(int i=0; i < ROI_num; ++i){
			num_id =  ROI_data[i * ROI_data_length_];
			w = ROI_info[i*ROI_info_length_ +1];
			h = ROI_info[i*ROI_info_length_ +2];
			const Dtype* cur_predicted  = ROI_predicted + i  ;
			int class_id = cur_predicted[0] - 1;
			LOG(INFO)<<"in roi "<<i<<" find class_id = "<<class_id;
			if(class_id >= 0){
				out_data[top[0]->offset(num_id,class_id ,h, w)] = 1;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+0,h, w)] = ROI_diff[i*4*num_class_ + class_id +0] + w - ROI_data[i * ROI_data_length_ +1 ] ;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+1,h, w)] = ROI_diff[i*4*num_class_ + class_id +1] + h - ROI_data[i * ROI_data_length_ +2 ];
				out_data[top[0]->offset(num_id,num_class_+class_id*4+2,h, w)] = ROI_diff[i*4*num_class_ + class_id +2] + w - ROI_data[i * ROI_data_length_ +3 ];;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+3,h, w)] = ROI_diff[i*4*num_class_ + class_id +3] + h - ROI_data[i * ROI_data_length_ +4 ];;
			}
		}
	}else if(label_type_ == ROI2HeatMapParam_LabelType_NDim){
		for(int i=0; i < ROI_num; ++i){
			num_id =  ROI_data[i * ROI_data_length_];
			w = ROI_info[i*ROI_info_length_ +1];
			h = ROI_info[i*ROI_info_length_ +2];
			const Dtype* cur_predicted  = ROI_predicted + i*(num_class_ ) ;
			for(int class_id = 0; class_id < num_class_; ++class_id){
				out_data[top[0]->offset(num_id,class_id ,h, w)] = cur_predicted[class_id];
				out_data[top[0]->offset(num_id,num_class_+class_id*4+0,h, w)] = ROI_diff[i*4*num_class_ + class_id +0] + w - ROI_data[i * ROI_data_length_ +1 ] ;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+1,h, w)] = ROI_diff[i*4*num_class_ + class_id +1] + h - ROI_data[i * ROI_data_length_ +2 ];
				out_data[top[0]->offset(num_id,num_class_+class_id*4+2,h, w)] = ROI_diff[i*4*num_class_ + class_id +2] + w - ROI_data[i * ROI_data_length_ +3 ];;
				out_data[top[0]->offset(num_id,num_class_+class_id*4+3,h, w)] = ROI_diff[i*4*num_class_ + class_id +3] + h - ROI_data[i * ROI_data_length_ +4 ];;
			}
		}
	}else
		LOG(FATAL) << "Unknown type " << label_type_;

}


template <typename Dtype>
void ROI2HeatMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(ROI2HeatMapLayer);
#endif



INSTANTIATE_CLASS(ROI2HeatMapLayer);
REGISTER_LAYER_CLASS(ROI2HeatMap);


}  // namespace caffe

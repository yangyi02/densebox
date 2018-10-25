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
void ROIRefineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	const ROIRefineParam& roi_refine_param = this->layer_param_.roi_refine_param();
	label_type_ = roi_refine_param.label_type();
	num_class_ = roi_refine_param.num_class();
	ROI_data_length_ = bottom[0]->width();
	ROI_info_length_ = bottom[1]->width();
	top[0]->Reshape(1, 1, 1, ROI_data_length_);
	top[1]->Reshape(1, 1, 1, ROI_info_length_);

	has_ROI_score_ = (bottom.size()>3);

}


template <typename Dtype>
void ROIRefineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	int num  = bottom[0]->num();
	CHECK_EQ(num,bottom[1]->num());

	CHECK_EQ(num,bottom[2]->num());
	CHECK_EQ(bottom[2]->width(),4*num_class_);

	if(has_ROI_score_){
		CHECK_EQ(num,bottom[3]->num());
		if(label_type_ == ROIRefineParam_LabelType_NPlus1)
			CHECK_EQ(bottom[3]->width(),num_class_ + 1);
		else if(label_type_ == ROIRefineParam_LabelType_OneDim)
			CHECK_EQ(bottom[3]->width(), 1);
		else if(label_type_ == ROIRefineParam_LabelType_NDim)
			CHECK_EQ(bottom[3]->width(), num_class_);
		else
			LOG(FATAL) << "Unknown type " << label_type_;
	}
	top[0]->Reshape(num, 1, 1, ROI_data_length_);
	top[1]->Reshape(num, 1, 1, ROI_info_length_);
}

/**
 * @brief Refine ROI output by ROI_diff and ROI_predicted
 *		  The input has at least two blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the bbox_diff and predicted labels.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 *
 *  @TODO a probability to change class_id to most likely class ?
 */
template <typename Dtype>
void ROIRefineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	int num  = bottom[0]->num();

	Dtype* top_ROI_data  = top[0]->mutable_cpu_data();
	Dtype* top_ROI_info  = top[1]->mutable_cpu_data();
	Dtype* bottom_ROI_diff = bottom[2]->mutable_cpu_data();
	Dtype* bottom_ROI_score = NULL;

	caffe::caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
	caffe::caffe_copy(bottom[1]->count(), bottom[1]->cpu_data(), top[1]->mutable_cpu_data());

	for(int i=0; i <num; ++i){
		Dtype* cur_top_ROI_data = top_ROI_data + i * ROI_data_length_ + 1;
		int class_id =  top_ROI_info[i * ROI_info_length_];
		Dtype* cur_bottom_ROI_diff = bottom_ROI_diff + i*4*num_class_ + class_id;
		*(cur_top_ROI_data ++ ) -= *(cur_bottom_ROI_diff++);
		*(cur_top_ROI_data ++ ) -= *(cur_bottom_ROI_diff++);
		*(cur_top_ROI_data ++ ) -= *(cur_bottom_ROI_diff++);
		*(cur_top_ROI_data ++ ) -= *(cur_bottom_ROI_diff++);
	}
	if(has_ROI_score_){
		// @TODO a probability to change class_id to most likely class ?
		bottom_ROI_score = bottom[3]->mutable_cpu_data();
		if(label_type_ == ROIRefineParam_LabelType_NPlus1){
			for(int i=0; i <num; ++i){
				int class_id =  top_ROI_info[i * ROI_info_length_];
				top_ROI_info[i * ROI_info_length_ + 3] = bottom_ROI_score[i * (num_class_ + 1)+1+class_id];
			}
		}
		else if(label_type_ == ROIRefineParam_LabelType_OneDim){
			for(int i=0; i <num; ++i){
				int class_id =  top_ROI_info[i * ROI_info_length_];
				top_ROI_info[i * ROI_info_length_ + 3] = ( (bottom_ROI_score[i] - 1) == class_id);
			}
		}
		else if(label_type_ == ROIRefineParam_LabelType_NDim){
			for(int i=0; i <num; ++i){
				int class_id =  top_ROI_info[i * ROI_info_length_];
				top_ROI_info[i * ROI_info_length_ + 3] = bottom_ROI_score[i * num_class_ +class_id];
			}
		}
		else
			LOG(FATAL) << "Unknown type " << label_type_;
	}

}


template <typename Dtype>
void ROIRefineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(ROIRefineLayer);
#endif



INSTANTIATE_CLASS(ROIRefineLayer);
REGISTER_LAYER_CLASS(ROIRefine);


}  // namespace caffe

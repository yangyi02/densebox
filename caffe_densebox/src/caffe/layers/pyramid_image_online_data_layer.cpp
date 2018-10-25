#include <string>
#include <vector>
#include <algorithm>
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/pyramid_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

namespace caffe {

template <typename Dtype>
int PyramidImageOnlineDataLayer<Dtype>::GetTotalSampleSize(){
	return 9999999;
}


template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::LoadOneImgToInternalBlob(const cv::Mat& img){
	if(Caffe::mode() == Caffe::GPU){
		LoadOneImgToInternalBlob_gpu(img);
	}else{
		LoadOneImgToInternalBlob_cpu(img);
	}
}

template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::LoadOneImgToInternalBlob_cpu(const cv::Mat& img){
	int buff_block_id = (this->used_buffered_block_id_+1)%(this->buffered_block_.size());
	Blob<Dtype>& img_blob_ = *(this->img_blob_list_[buff_block_id]);
	ReadImgToBlob(img, img_blob_, this->mean_bgr_[0],this->mean_bgr_[1],this->mean_bgr_[2]);
}


template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::LoadSampleToImgBlob(
		pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_){

	cur_sample_.first = string("Image in memorry");
	CHECK_GT(img_blob_.width(),0);
	CHECK_GT(img_blob_.height(),0);
	CHECK_GT(img_blob_.channels(),0);
	CHECK_GT(img_blob_.num(),0);

}

template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::SetRoiAndScale(RectBlockPacking<Dtype>& rect_block_packer_,
		pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_){
	/**
	 * set scales for patchwork.
	 */
	if(this->scale_from_annotation_ == false){
		vector<Dtype> scales ;
		scales.clear();
		int num_scale = (this->scale_end_ - this->scale_start_)/this->scale_step_;
		if(num_scale > 30)
			LOG(INFO)<<"Warning, too much testing scales: "<< num_scale;
		for(Dtype scale = this->scale_start_; scale <= this->scale_end_; scale += this->scale_step_){
			Dtype cal_scale = pow(2,scale);
			scales.push_back(cal_scale);
		}
		if (this->pic_print_)
			LOG(INFO)<<"img_blob height:"<<img_blob_.height()<<"  width: "<<img_blob_.width();
		rect_block_packer_.setRoi(img_blob_, scales);
	}else{
		rect_block_packer_.setRoi(img_blob_, cur_sample_);
		cur_sample_.second.clear();
	}


}



template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::StartProcessOneImg(){
	this->InternalThreadEntry();
	this->used_buffered_block_id_ = (this->used_buffered_block_id_+1)%(this->buffered_block_.size());
	this->forward_times_for_cur_sample_ = std::ceil((
			this->buffered_block_[this->used_buffered_block_id_])->num()
			/(this->max_block_num_+0.0));
	this->cur_sample_id_ = (this->cur_sample_id_+1)/GetTotalSampleSize();

}

template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::SetROIWithScale(const vector<ROIWithScale>& roi_scale){
	int buff_block_id = (this->used_buffered_block_id_+1)%(this->buffered_block_.size());
	pair<string, vector<Dtype> >&  cur_sample_ = *(this->cur_sample_list_[buff_block_id]);
	vector<Dtype>& roi_info = cur_sample_.second;
	roi_info.resize(roi_scale.size()*5,0);
	for(int i=0; i < roi_scale.size(); ++i){
		int roi_info_idx = i*5;
		const ROIWithScale& cur_roi = roi_scale[i];
		roi_info[roi_info_idx] = cur_roi.l;
		roi_info[roi_info_idx + 1] = cur_roi.t;
		roi_info[roi_info_idx + 2] = cur_roi.r;
		roi_info[roi_info_idx + 3] = cur_roi.b;
		if(cur_roi.scale > 0.02){
			roi_info[roi_info_idx + 4] = std::log10(cur_roi.scale)/std::log10(2);
		}else{
			LOG(INFO)<<"warning, scale is too small: "<<cur_roi <<" and use scale = 1 instead. ";
			roi_info[roi_info_idx + 4] = 0;
		}
	}

}


#ifdef CPU_ONLY
STUB_GPU(PyramidImageOnlineDataLayer);
#endif


INSTANTIATE_CLASS(PyramidImageOnlineDataLayer);
REGISTER_LAYER_CLASS(PyramidImageOnlineData);



}  // namespace caffe

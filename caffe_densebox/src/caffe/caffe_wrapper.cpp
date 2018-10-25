/*
 * caffe_wrapper.cpp
 *
 *      Author: Alan_Huang
 */
#include "caffe/blob.hpp"
#include "caffe/caffe_wrapper.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/pyramid_data_layers.hpp"

#include "caffe/net.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/common.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


namespace caffe {
using std::vector;

CaffeDenseBoxDetector::CaffeDenseBoxDetector(const std::string proto_name,
		const std::string model_name,
		const bool use_cuda  ){
	net_ = NULL;
	SetCudaFlag(use_cuda);
	Net<float>* net_ptr_ = new Net<float>(proto_name,caffe::TEST);
	net_ptr_->CopyTrainedLayersFrom(model_name);
	net_ = static_cast<void*>(net_ptr_);
}

CaffeDenseBoxDetector::CaffeDenseBoxDetector(const std::string proto_name,
		const bool use_cuda  ){
	net_ = NULL;
	SetCudaFlag(use_cuda);
	Net<float>* net_ptr_ = new Net<float>(proto_name,caffe::TEST);
	net_ = static_cast<void*>(net_ptr_);
}


CaffeDenseBoxDetector::~CaffeDenseBoxDetector(){
	if(net_){
		Net<float>* net_ptr = static_cast<Net<float>* >(net_);
		if(net_ptr != NULL)
			delete net_ptr;
		net_ = NULL;
	}
}


void CaffeDenseBoxDetector::SetCudaFlag(bool flag){
	if(flag == false){
		Caffe::set_mode(Caffe::CPU);
	}else{
		Caffe::set_mode(Caffe::GPU);
	}
}

void CaffeDenseBoxDetector::CopyFromModel(const std::string model_name){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	net_ptr->CopyTrainedLayersFrom(model_name);
}

bool CaffeDenseBoxDetector::LoadImgToBuffer(cv::Mat & src_img){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	if(src_img.data){
		PyramidImageOnlineDataLayer<float>* input_data_layer =
			 static_cast<PyramidImageOnlineDataLayer<float>*>(net_ptr->layers()[0].get());
		input_data_layer->LoadOneImgToInternalBlob(src_img);
		return true;
	}else{
		LOG(INFO)<<" src_img(cv::Mat) is invalid";
		return false;
	}

}
bool CaffeDenseBoxDetector::SetRoiWithScale(const vector<ROIWithScale>& roi_scale){
	if(roi_scale.size() < 1){
		LOG(INFO)<<"roi_scale is empty! ";
		return false;
	}
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	PyramidImageOnlineDataLayer<float>* input_data_layer =
		 static_cast<PyramidImageOnlineDataLayer<float>*>(net_ptr->layers()[0].get());
	input_data_layer->SetROIWithScale(roi_scale);
	return true;
}

void CaffeDenseBoxDetector::PredictOneImg(){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	PyramidImageOnlineDataLayer<float>* input_data_layer =
		 static_cast<PyramidImageOnlineDataLayer<float>*>(net_ptr->layers()[0].get());
	net_ptr->ForwardPrefilled();
	int forward_times_for_this_sample = input_data_layer->GetForwardTimesForCurSample();
	for(int iter = 1; iter<forward_times_for_this_sample; ++iter ){
		net_ptr->ForwardPrefilled();
	}

}

int CaffeDenseBoxDetector::ClassNum(){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	DetectionOutputLayer<float>* output_data_layer =
		static_cast<DetectionOutputLayer<float>*>(net_ptr->layers()[net_ptr->layers().size()-1].get());
	return output_data_layer->GetNumClass();
}

vector< BBox<float> >& CaffeDenseBoxDetector::GetBBoxResults(int class_id){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	DetectionOutputLayer<float>* output_data_layer =
			static_cast<DetectionOutputLayer<float>*>(net_ptr->layers()[net_ptr->layers().size()-1].get());
	int class_num = output_data_layer->GetNumClass();
	CHECK_LT(class_id, class_num);
	return output_data_layer->GetFilteredBBox(class_id);
}

std::vector<std::vector< BBox<float> > >& CaffeDenseBoxDetector::GetBBoxResults(){
	Net<float>* net_ptr = static_cast<Net<float>* >(net_);
	DetectionOutputLayer<float>* output_data_layer =
				static_cast<DetectionOutputLayer<float>*>(net_ptr->layers()[net_ptr->layers().size()-1].get());
	return output_data_layer->GetFilteredBBox();
}

}

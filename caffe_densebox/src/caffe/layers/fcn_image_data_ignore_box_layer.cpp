#include <string>
#include <vector>

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
namespace caffe {

template <typename Dtype>
void IImageDataIgnoreBox<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_image_data_param);
	CHECK(fcn_image_data_param.has_fcn_image_data_ignore_box_param())<<
		"fcn_image_data_ignore_box_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_ignore_box_param());
}

/**
 * TODO
 */
template <typename Dtype>
int IImageDataIgnoreBox<Dtype>::SetUpChannelInfo( const int channel_base_offset  )
{
	return 0;
}

template <typename Dtype>
void IImageDataIgnoreBox<Dtype>::SetUpParameter(
		const FCNImageDataIgnoreBoxParameter& fcn_img_data_ignore_box_param)
{
	this->ignore_box_flag_id_ = fcn_img_data_ignore_box_param.ignore_box_flag_id();
	this->ignore_box_point_id_.clear();
	std::copy(fcn_img_data_ignore_box_param.ignore_box_point_id().begin(),
				fcn_img_data_ignore_box_param.ignore_box_point_id().end(),
	  	        std::back_inserter(this->ignore_box_point_id_));
	CHECK(this->ignore_box_point_id_.size() == 2)<<"this->ignore_box_id_.size() == "<<
				this->ignore_box_point_id_.size()<<" bbox point number should be 2.";
	LOG(INFO)<<"need_ignore_box_flag is activated. ignore_box_flag_id  = " <<fcn_img_data_ignore_box_param.ignore_box_flag_id();;
}

template <typename Dtype>
void IImageDataIgnoreBox<Dtype>::GenerateIgnoreBoxMap(
		int item_id, vector<float> & coords,vector<float> & box_scale,const LayerParameter& param)
{

}


/**
 * TODO
 */
template <typename Dtype>
void IImageDataIgnoreBox<Dtype>:: PrintPic(int item_id, const string & output_path,
		cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label)
{

}

INSTANTIATE_CLASS(IImageDataIgnoreBox);
}  // namespace caffe

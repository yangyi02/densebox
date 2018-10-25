#include <string>
#include <vector>
#include <sys/param.h>
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_others.hpp"
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {


template <typename Dtype>
void IImageDataProcessor<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	CHECK(fcn_image_data_param.has_fcn_image_data_common_param())<<
			"fcn_image_data_common_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_common_param());
}
template <typename Dtype>
void IImageDataProcessor<Dtype>::SetUpParameter(
		const FCNImageDataCommonParameter& fcn_img_data_common_param)
{

	this->input_height_ = fcn_img_data_common_param.input_height();
	this->input_width_ = fcn_img_data_common_param.input_width();
	this->heat_map_a_ = fcn_img_data_common_param.heat_map_a();
	this->heat_map_b_ = fcn_img_data_common_param.heat_map_b();
	this->out_height_ = fcn_img_data_common_param.out_height();
	this->out_width_ = fcn_img_data_common_param.out_width();
	this->single_thread_ = fcn_img_data_common_param.single_thread();

	this->scale_positive_lower_bounder_ = fcn_img_data_common_param.scale_positive_lower_bounder();
	this->scale_positive_upper_bounder_ = fcn_img_data_common_param.scale_positive_upper_bounder();
	this->scale_ignore_lower_bounder_ = fcn_img_data_common_param.scale_ignore_lower_bounder();
	this->scale_ignore_upper_bounder_ = fcn_img_data_common_param.scale_ignore_upper_bounder();
	scale_ignore_lower_bounder_ = MIN(scale_ignore_lower_bounder_,scale_positive_lower_bounder_);
	scale_ignore_upper_bounder_ = MAX(scale_ignore_upper_bounder_,scale_positive_upper_bounder_);

	scale_choose_stragety_ = fcn_img_data_common_param.scale_choose_strategy();
	this->scale_base_.clear();
	std::copy(fcn_img_data_common_param.scale_base().begin(),
			fcn_img_data_common_param.scale_base().end(),
			std::back_inserter(this->scale_base_));
	if(scale_base_.size() == 0)
	{
		this->scale_base_.push_back(1);
	}
	SetScaleSamplingWeight();
	CHECK(fcn_img_data_common_param.has_num_anno_points_per_instance());
	this->num_anno_points_per_instance_ =
			fcn_img_data_common_param.num_anno_points_per_instance();

	show_output_path_ = string("cache/ImageMultiScaleDataLayer");

	pic_print_ = ((getenv("PIC_PRINT") != NULL) && (getenv("PIC_PRINT")[0] == '1'));
	label_print_ = (getenv("LABEL_PRINT") != NULL) && (getenv("LABEL_PRINT")[0] == '1');;
	if (pic_print_ || label_print_) {
		CreateDir(show_output_path_.c_str());
	}

	char output_path[512];
	if (this->pic_print_) {
		sprintf(output_path, "%s/pic",this->show_output_path_.c_str());
		CreateDir(output_path);
	}
	if (this->label_print_) {
		sprintf(output_path, "%s/label",this->show_output_path_.c_str());
		CreateDir(output_path);
	}

	PIC_MARGIN = 10;
}

template <typename Dtype>
void IImageDataProcessor<Dtype>::SetScaleSamplingWeight(){
	scale_sampling_weight_.clear();
	Dtype sum = 0;
	for(int i=0; i < scale_base_.size();++i){
		CHECK_GT(scale_base_[i],0);
		scale_sampling_weight_.push_back(1/scale_base_[i]);
		sum += scale_sampling_weight_[i];
	}
	std::ostringstream oss;
	oss<<"The scale_sampling_weights are: [";
	for(int i=0; i < scale_base_.size();++i){
		scale_sampling_weight_[i] /= sum;
		oss<<scale_sampling_weight_[i]<<" ";
	}
	oss<<"].";
	LOG(INFO)<<oss.str();
}
template <typename Dtype>
int IImageDataProcessor<Dtype>::GetWeighedScaleIdByCDF(Dtype point){
	CHECK_GE(point, 0);
	CHECK_LE(point, 1);
	Dtype cdf = 0;
	for(int i=0; i < scale_sampling_weight_.size(); ++i){
		cdf += scale_sampling_weight_[i];
		if(cdf >= point){
			return i;
		}
	}
	LOG(ERROR)<<"In GetWeighedScaleIdByCDF(), the input "<<point<<" exceed 1.";
	return 0;
}

template <typename Dtype>
string IImageDataProcessor<Dtype>::GetPrintSampleName(const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type)
{
	string out_sample_name;
	std::vector<std::string> splited_name= std_split(cur_sample.first,"/");
	switch(sample_type)
	{
		case caffe::SOURCE_TYPE_POSITIVE: {
			out_sample_name = string("pos_") + splited_name[splited_name.size()-1];
			break;
		}
		case caffe::SOURCE_TYPE_ALL_NEGATIVE: {
			out_sample_name = string("all_neg_") + splited_name[splited_name.size()-1];
			break;
		}
		case caffe::SOURCE_TYPE_HARD_NEGATIVE: {
			out_sample_name = string("harg_neg") + splited_name[splited_name.size()-1];
			break;
		}
		case caffe::SOURCE_TYPE_POSITIVE_WITH_ROI: {
			out_sample_name = string("roi_") + splited_name[splited_name.size()-1];
			break;
		}
		default:
			LOG(FATAL) << "Unknown type " << sample_type;
	}
	return out_sample_name;
}


template <typename Dtype>
vector<ImageDataAnnoType> IImageDataProcessor<Dtype>::GetAnnoTypeForAllScaleBase( Dtype scale)
{
	vector<ImageDataAnnoType> anno_types;
	int n_scales = this->scale_base_.size();
	for(int i=0; i < n_scales; ++i)
	{
		if(scale < this->scale_base_[i] * this->scale_ignore_lower_bounder_)
		{
			anno_types.push_back(caffe::ANNO_NEGATIVE);
		}
		else if(scale <= this->scale_base_[i] * this->scale_positive_lower_bounder_)
		{
			anno_types.push_back(caffe::ANNO_IGNORE);
		}
		else if(scale <= this->scale_base_[i] * this->scale_positive_upper_bounder_)
		{
			anno_types.push_back(caffe::ANNO_POSITIVE);
		}
		else if(scale <= this->scale_base_[i] * this->scale_ignore_upper_bounder_)
		{
			anno_types.push_back(caffe::ANNO_IGNORE);
		}
		else
		{
			anno_types.push_back(caffe::ANNO_NEGATIVE);
		}
		if (this->pic_print_ || this->label_print_)
			LOG(INFO)<<"			detection scale: "<<scale<<" anno_type: "<< anno_types[anno_types.size()-1];
	}
	CHECK_EQ(anno_types.size(),n_scales);
	return anno_types;
}


template <typename Dtype>
IImageDataBoxNorm<Dtype>::IImageDataBoxNorm(){
	bbox_height_ = bbox_width_ = 0;
	bbox_size_norm_type_ = FCNImageDataBoxNormParameter_BBoxSizeNormType_HEIGHT;
}

template <typename Dtype>
IImageDataBoxNorm<Dtype>::~IImageDataBoxNorm(){

}

/**
 * @todo
 */
template <typename Dtype>
void IImageDataBoxNorm<Dtype>::SetUpParameter(const FCNImageDataParameter& fcn_image_data_param){

}

/**
 * @todo
 */
template <typename Dtype>
vector<Dtype> IImageDataBoxNorm<Dtype>::GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance,
		int num_points_per_instance, vector<int>& bbox_point_ids){
	vector<Dtype> a;
	return a;
}

/**
 * @todo
 */
template <typename Dtype>
vector<ImageDataAnnoType> IImageDataBoxNorm<Dtype>::GetAnnoTypeForAllScaleBase( Dtype scale){
	vector<ImageDataAnnoType> a;
	return a;
}

INSTANTIATE_CLASS(IImageDataBoxNorm);
INSTANTIATE_CLASS(IImageDataProcessor);
}  // namespace caffe

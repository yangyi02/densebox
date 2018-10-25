#include <string>
#include <vector>

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_others.hpp"
namespace caffe {


template <typename Dtype>
FCNImageDataLayer<Dtype>::~FCNImageDataLayer<Dtype>() {
	this->StopInternalThread();
}

template <typename Dtype>
void FCNImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	BaseImageDataLayer<Dtype>::DataLayerSetUp(bottom,top);
	CHECK_EQ(bottom.size(), 0)<< "Input Layer takes no input blobs.";
	CHECK_GE(top.size(), 1) << "Input Layer takes at least one blobs as output.";


	CHECK(this->layer_param_.has_fcn_image_data_param());
	SetUpParameter(this->layer_param_.fcn_image_data_param());
	SetUpChannelInfo(0);

	if(this->need_detection_box_ || this->need_keypoint_){
		CHECK_GT(top.size(), 1) << "Input Layer takes at least two blobs as output in detection task.";
	}

	int c_dim = 3  + this->is_img_pair_ *3;

	top[0]->Reshape(this->batch_size_, c_dim, this->input_height_,this->input_width_);
	for(int i=0; i < this->PREFETCH_COUNT; ++i){
		this->prefetch_[i].data_.ReshapeLike(*(top[0]));
	}

	LOG(INFO) << "output data size: " << (top)[0]->num() << ","
	  << (top)[0]->channels() << "," << (top)[0]->height() << ","
	  << (top)[0]->width();


	if(this->total_channel_need_ > 0 && top.size()>1){
		top[1]->Reshape(this->batch_size_,this->total_channel_need_, this->out_height_,this->out_width_);
		for(int i=0; i < this->PREFETCH_COUNT; ++i){
			this->prefetch_[i].label_.ReshapeLike(*(top[1]));
		}
		LOG(INFO) << "output label size: " << (top)[1]->num() << ","
		  << (top)[1]->channels() << "," << (top)[1]->height() << ","
		  << (top)[1]->width();
	}

	if(this->phase_ == TEST){
		data_provider_.SetNegRatio(0);
		LOG(INFO)<<"In Testing phase, the neg_ratio in data provider is set to 0.";
	}
}

template <typename Dtype>
int FCNImageDataLayer<Dtype>::SetUpChannelInfo( const int channel_base_offset  )
{
	IImageDataProcessor<Dtype>::SetUpChannelInfo(channel_base_offset);
	if(need_detection_box_)
	{
		IImageDataDetectionBox<Dtype>::SetUpChannelInfo(this->total_channel_need_);
	}
	if(need_keypoint_)
	{
		IImageDataKeyPoint<Dtype>::SetUpChannelInfo(this->total_channel_need_);
	}
	if(need_detection_box_)
	{
		IImageDataIgnoreBox<Dtype>::SetUpChannelInfo(this->total_channel_need_);
	}
	return this->total_channel_need_;
}

template <typename Dtype>
void FCNImageDataLayer<Dtype>::SetUpParameter(const FCNImageDataParameter & fcn_img_data_param)
{

	CHECK(fcn_img_data_param.has_fcn_image_data_common_param());
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_img_data_param.fcn_image_data_common_param());

	IImageBufferedDataReader<Dtype>::SetUpParameter(fcn_img_data_param );


	need_detection_box_ = fcn_img_data_param.has_fcn_image_data_detection_box_param();
	if(need_detection_box_)
	{
		IImageDataDetectionBox<Dtype>::SetUpParameter(fcn_img_data_param.fcn_image_data_detection_box_param());
	}
	need_keypoint_ = fcn_img_data_param.has_fcn_image_data_key_point_param();
	if(need_keypoint_)
	{
		IImageDataKeyPoint<Dtype>::SetUpParameter(fcn_img_data_param.fcn_image_data_key_point_param());
	}
	need_ignore_box_ = fcn_img_data_param.has_fcn_image_data_ignore_box_param();
	if(need_ignore_box_)
	{
		IImageDataIgnoreBox<Dtype>::SetUpParameter(fcn_img_data_param.fcn_image_data_ignore_box_param());
	}
	data_provider_.SetUpParameter(fcn_img_data_param);
	data_provider_.ReadPosAndNegSamplesFromFiles(fcn_img_data_param.fcn_image_data_source_param(),
			this->num_anno_points_per_instance_,this->class_flag_id_);
	this->batch_size_ = data_provider_.GetBatchSize() ;
	for(int i=0; i < this->batch_size_; ++i){
		cv_img_.push_back(cv::Mat());
		cv_img_depth_.push_back(cv::Mat());
		cv_img_original_.push_back(cv::Mat());
		cv_img_original_depth_.push_back(cv::Mat());
	}
}


template <typename Dtype>
void FCNImageDataLayer<Dtype>::BatchSetUp(Batch<Dtype>* batch)
{
	data_provider_.FetchBatchSamples();
	vector<std::pair<std::string, vector<Dtype> > > & buff_batch_samples_ =
	  		batch->buff_batch_samples_;
	buff_batch_samples_.clear();

	for(int i=0; i < this->batch_size_; ++i){
		buff_batch_samples_.push_back(this->data_provider_.GetMutableSampleInBatchAt(i));
	}
	if(this->total_channel_need_ > 0 && batch->label_.count()>0){
		caffe::caffe_set(batch->label_.count(),Dtype(0),batch->label_.mutable_cpu_data());
	}
	caffe::caffe_set(batch->data_.count(),Dtype(0),batch->data_.mutable_cpu_data());
	this->gt_bboxes_.clear();
	if(chosen_scale_id_.size()>0){
		vector<Dtype> chosen_scale_count(this->scale_base_.size(),0);
		for(int i=0 ; i < this->chosen_scale_id_.size(); ++i){
			chosen_scale_count[chosen_scale_id_[i]]+= 1.0/ this->chosen_scale_id_.size() ;
		}
		std::ostringstream oss;
		oss<<"					The frequency of each scale_base_id in previous batch: [";
		for(int i=0; i < this->scale_base_.size();++i){
			oss<<chosen_scale_count[i]<<" ";
		}
		oss<<"].";
//		LOG(INFO)<<oss.str();
	}
	chosen_scale_id_.clear();

}

template <typename Dtype>
void FCNImageDataLayer<Dtype>::BatchFinalize(Batch<Dtype>* batch){
	if(this->need_prefetch_bbox_){
		this->GTBBoxesToBlob(batch->bbox_);
		if(this->bbox_print_){
			IImageDataDetectionBox<Dtype>::PrintBBoxes(batch->bbox_);
		}
	}

}

template <typename Dtype>
int FCNImageDataLayer<Dtype>::GetScaleBaseId(){
	int returned_scale_base_id = 0;
	switch(this->scale_choose_stragety_){
		case FCNImageDataCommonParameter_ScaleChooseStrategy_RANDOM:{
			returned_scale_base_id = this->PrefetchRand() % this->scale_base_.size();
			break;
		}
		case FCNImageDataCommonParameter_ScaleChooseStrategy_WEIGHTED:{
			returned_scale_base_id = this->GetWeighedScaleIdByCDF(this->PrefetchRandFloat());
			break;
		}
		default:
			LOG(FATAL) << "Unknown scale choose stragety " << this->scale_choose_stragety_;
	}
	return returned_scale_base_id;
}


template <typename Dtype>
vector<bool>  FCNImageDataLayer<Dtype>::SetPointTransformIgnoreFlag(const std::pair<std::string, vector<Dtype> >& cur_sample,
		ImageDataSourceSampleType sample_type){
	vector<bool> is_keypoint_transform_ignored(cur_sample.second.size(), false);
	int num_instances =  cur_sample.second.size() / (this->num_anno_points_per_instance_ * 2);
	if(this->ignore_class_flag_id_ > -1  && (sample_type == caffe::SOURCE_TYPE_POSITIVE||
			sample_type == caffe::SOURCE_TYPE_POSITIVE_WITH_ROI)){
		CHECK_LT(this->ignore_class_flag_id_,this->num_anno_points_per_instance_);

		CHECK(cur_sample.second.size() % (this->num_anno_points_per_instance_ * 2) == 0);
		for(int inst_id = 0 ; inst_id < num_instances; ++inst_id){
			is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + this->ignore_class_flag_id_*2 ] = true;
			is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + this->ignore_class_flag_id_*2 +1 ] = true;
		}
	}

	if((sample_type == caffe::SOURCE_TYPE_POSITIVE|| sample_type == caffe::SOURCE_TYPE_POSITIVE_WITH_ROI)
			&& this->total_class_num_ > 1 && this->need_detection_box_){
//		LOG(INFO)<<"this->total_class_num_:   				"<<this->total_class_num_;

		CHECK(cur_sample.second.size() % (this->num_anno_points_per_instance_ * 2) == 0);
		for(int inst_id = 0 ; inst_id < num_instances; ++inst_id){
			is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + this->class_flag_id_*2 ] = true;
			is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + this->class_flag_id_*2 +1 ] = true;
		}
	}
	if((sample_type == caffe::SOURCE_TYPE_POSITIVE|| sample_type == caffe::SOURCE_TYPE_POSITIVE_WITH_ROI)
			&& this->need_keypoint_){
		CHECK(cur_sample.second.size() % (this->num_anno_points_per_instance_ * 2) == 0);
		for(int i=0; i < this->ignore_key_point_flag_idxs_.size(); ++i){
			int ignore_point_flag_idx = this->ignore_key_point_flag_idxs_[i];
			for(int inst_id = 0 ; inst_id < num_instances; ++inst_id){
				is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + ignore_point_flag_idx*2 ] = true;
				is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + ignore_point_flag_idx*2 +1] = true;
			}
		}

		for(int i=0; i < this->used_attribute_point_idxs_.size();++i){
			int attribute_point_idx = this->used_attribute_point_idxs_[i];
			for(int inst_id = 0 ; inst_id < num_instances; ++inst_id){
				is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + attribute_point_idx*2 ] = true;
				is_keypoint_transform_ignored[inst_id *this->num_anno_points_per_instance_ * 2 + attribute_point_idx*2 +1 ] = true;
			}
		}


	}
	return is_keypoint_transform_ignored;
}

template <typename Dtype>
void FCNImageDataLayer<Dtype>::ProcessImg(Batch<Dtype>* batch,int item_id)
{

	std::pair<std::string, vector<Dtype> >   cur_sample =  data_provider_.GetMutableSampleInBatchAt(item_id);
	ImageDataSourceSampleType  sample_type = data_provider_.GetMutableSampleTypeInBatchAt(item_id);
	/**
	 * re-organize the annotation point if sample type is SOURCE_TYPE_POSITIVE_WITH_ROI.
	 */
	if(sample_type == caffe::SOURCE_TYPE_POSITIVE_WITH_ROI){

//		std::cout<<"before resize, coord:";
//		for(int i=0; i < cur_sample.second.size(); ++i){
//			std::cout<<cur_sample.second[i]<<" ";
//		}
//		std::cout<< std::endl;;
		const int num_roi_points_per_instance = ImageDataROISourceProvider<Dtype>::num_roi_points_per_instance_;
//		LOG(INFO)<<"		cur_sample.second.size():"<<cur_sample.second.size();
		CHECK_EQ( (cur_sample.second.size() - num_roi_points_per_instance*2)
				% (this->num_anno_points_per_instance_*2), 0);

		vector<Dtype> previous_points =cur_sample.second;
		vector<Dtype>& sample_points = cur_sample.second;
		CHECK_GE(previous_points.size(), num_roi_points_per_instance*2);

		sample_points = vector<Dtype>((cur_sample.second.size() - num_roi_points_per_instance*2)
				+ this->num_anno_points_per_instance_*2 ,-1);
		copy(previous_points.begin() +num_roi_points_per_instance*2, previous_points.end(),
				sample_points.begin()+this->num_anno_points_per_instance_*2);
		sample_points[this->standard_len_point_1_*2] = previous_points[0];
		sample_points[this->standard_len_point_1_*2+1] = previous_points[1];
		sample_points[this->standard_len_point_2_*2] = previous_points[2];
		sample_points[this->standard_len_point_2_*2+1] = previous_points[3];
		sample_points[this->roi_center_point_*2] = previous_points[4];
		sample_points[this->roi_center_point_*2+1] = previous_points[5];

//		std::cout<<"after resize, coord:";
//		for(int i=0; i < cur_sample.second.size(); ++i){
//			std::cout<<cur_sample.second[i]<<" ";
//		}
//		std::cout<< std::endl;;
//		CHECK(false);
//		LOG(INFO)<<"		after resize, cur_sample.second.size():"<<cur_sample.second.size();
	}


	vector<bool> is_keypoint_transform_ignored = SetPointTransformIgnoreFlag(cur_sample,sample_type);
	/**
	 * randomly choose scale_base_id;
	 */
	int used_scale_base_id = GetScaleBaseId();
	boost::unique_lock<boost::shared_mutex> write_lock(this->mutex_);
	chosen_scale_id_.push_back(used_scale_base_id);
	write_lock.unlock();


	/**
	 *  pre-process Image
	 */
	cv::Mat& cv_img = cv_img_[item_id];
	cv::Mat& cv_img_depth= cv_img_depth_[item_id];
	cv::Mat& cv_img_original= cv_img_original_[item_id];
	cv::Mat& cv_img_original_depth= cv_img_original_depth_[item_id];
	vector<cv::Mat*>  img_cv_ptrs;
	img_cv_ptrs.push_back(& cv_img);
	img_cv_ptrs.push_back(& cv_img_depth);
	vector<cv::Mat*>  img_ori_ptrs;
	img_ori_ptrs.push_back(& cv_img_original);
	img_ori_ptrs.push_back(& cv_img_original_depth);

	bool valid_source = IImageBufferedDataReader<Dtype>::ReadImgAndTransform( item_id,
			batch->data_,img_cv_ptrs, img_ori_ptrs,
			cur_sample, is_keypoint_transform_ignored,
			sample_type,
			this->phase_,used_scale_base_id);

	if(! valid_source) return;

	string output_path("cache/ImageMultiScaleDataLayer");

	if(this->pic_print_)
	{
		IImageDataReader<Dtype>::PrintPic(item_id, this->show_output_path_,& cv_img, & cv_img_original,cur_sample,
				sample_type,this->scale_base_[used_scale_base_id],batch->data_);
		if(this->is_img_pair_){
			IImageDataReader<Dtype>::PrintPic(item_id, this->show_output_path_,& cv_img_depth,
					& cv_img_original_depth,cur_sample, sample_type,
					this->scale_base_[used_scale_base_id],batch->data_,this->img_pair_postfix_);
		}
	}

	if(need_detection_box_){
		this->GenerateDetectionMap(item_id,cur_sample.second,batch->label_,used_scale_base_id);
		if(this->pic_print_){
				IImageDataDetectionBox<Dtype>::PrintPic(item_id, this->show_output_path_,& cv_img, & cv_img_original,cur_sample,
					sample_type,IImageDataReader<Dtype>::GetRefinedBaseScale(item_id),batch->label_);
		}
		if(this->label_print_){
			IImageDataDetectionBox<Dtype>::PrintLabel(item_id, this->show_output_path_,cur_sample,
					sample_type,IImageDataReader<Dtype>::GetRefinedBaseScale(item_id),batch->label_);
		}

	}
	if(this->need_keypoint_){
		this->GenerateKeyPointHeatMap(item_id,cur_sample.second,batch->label_,used_scale_base_id);
		if(this->pic_print_){
			IImageDataKeyPoint<Dtype>::PrintPic(item_id, this->show_output_path_,& cv_img, & cv_img_original,cur_sample,
					sample_type,IImageDataReader<Dtype>::GetRefinedBaseScale(item_id),batch->label_);
		}
	}

//	CHECK(false);
}


template <typename Dtype>
void FCNImageDataLayer<Dtype>::ShowDataAndPredictedLabel(const string & output_path, const Blob<Dtype>& data,
	const Blob<Dtype>& label,const Blob<Dtype>& predicted, Dtype threshold)
{
	char name[256];


	for(int n=0; n < data.num();++n){
		pair<string, vector<Dtype> > cur_sample = this->last_batch_samples_[n];
		vector<string> splited_name= std_split(cur_sample.first,"/");
		string img_name = splited_name[splited_name.size()-1];
		if(cur_sample.second.size() == 0){
			cur_sample.second.resize(this->num_anno_points_per_instance_ * 2,-1);
		}

		if(need_detection_box_){
			int bbox_id_1 = this->bbox_point_id_[0];
			int bbox_id_2 = this->bbox_point_id_[1];
			sprintf(name,"%s_%d_%d_%d_%d",img_name.c_str(),int(cur_sample.second[bbox_id_1*2]),int(cur_sample.second[bbox_id_1*2+1]),
					int(cur_sample.second[bbox_id_2*2]),int(cur_sample.second[bbox_id_2*2+1]));
			img_name = string(name);
			LOG(INFO)<<"saving result of detection image "<<img_name;
			IImageDataDetectionBox<Dtype>::ShowDataAndPredictedLabel( output_path,  img_name,
					  data, n,this->mean_bgr_,label, predicted, threshold);
		}
		if(this->need_keypoint_){
			int bbox_id_1 = this->key_point_standard_point_id1_ ;
			int bbox_id_2 = this->key_point_standard_point_id2_;
			sprintf(name,"%s_%d_%d_%d_%d",img_name.c_str(),int(cur_sample.second[bbox_id_1*2]),int(cur_sample.second[bbox_id_1*2+1]),
					int(cur_sample.second[bbox_id_2*2]),int(cur_sample.second[bbox_id_2*2+1]));
			img_name = string(name);
			LOG(INFO)<<"saving result of keypoint image "<<img_name;
			IImageDataKeyPoint<Dtype>::ShowDataAndPredictedLabel( output_path,  img_name,
					  data, n,this->mean_bgr_,label, predicted, threshold);
		}
	}

}


INSTANTIATE_CLASS(FCNImageDataLayer);
REGISTER_LAYER_CLASS(FCNImageData);
}  // namespace caffe

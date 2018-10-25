#include <string>
#include <vector>
#include <cmath>
#include "caffe/util/rng.hpp"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/common.hpp"
#include "caffe/util/buffered_reader.hpp"
namespace caffe {

/**
 * for IImageBufferedDataReader
 */
template <typename Dtype>
IImageBufferedDataReader<Dtype>::IImageBufferedDataReader():IImageDataReader<Dtype>()
{
	buffer_size_ = 25;
	is_img_pair_ = false;
	use_gpu_ = false;
	need_buffer_ = true;
	is_video_img_ = false;
	mean2_bgr_[0] = mean2_bgr_[1]=mean2_bgr_[2];

}

template <typename Dtype>
IImageBufferedDataReader<Dtype>::~IImageBufferedDataReader()
{
	buffered_reader_.reset();
}

template <typename Dtype>
void IImageBufferedDataReader<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_image_data_param);
	CHECK(fcn_image_data_param.has_fcn_image_data_reader_param())<<
			"fcn_image_data_reader_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_reader_param());
}

template <typename Dtype>
void IImageBufferedDataReader<Dtype>::SetUpParameter(
		const FCNImageDataReaderParameter& fcn_img_data_reader_param)
{
	IImageDataReader<Dtype>::SetUpParameter(fcn_img_data_reader_param);
	this->restrict_roi_in_center_ = fcn_img_data_reader_param.restrict_roi_in_center() && this->need_buffer_;
	this->is_img_pair_ = fcn_img_data_reader_param.is_img_pair();
	this->is_video_img_ = fcn_img_data_reader_param.is_video_img();
	this->use_gpu_ = fcn_img_data_reader_param.use_gpu();
	this->need_buffer_ = fcn_img_data_reader_param.need_buffer();
	this->mean2_bgr_[0] = fcn_img_data_reader_param.mean2_b();
	this->mean2_bgr_[1] = fcn_img_data_reader_param.mean2_g();
	this->mean2_bgr_[2] = fcn_img_data_reader_param.mean2_r();
	buff_blob_1_.clear();
	buff_blob_2_.clear();
	if(is_img_pair_){
		this->img_pair_postfix_ = fcn_img_data_reader_param.img_pair_postfix();
		buffered_reader_.reset(new BufferedColorJPGPairReader<Dtype>(img_pair_postfix_,buffer_size_));
		LOG(INFO)<<"In IImageBufferedDataReader, BufferedColorJPGPairReader is used";
	}
	else if(is_video_img_ ){
		if(fcn_img_data_reader_param.has_video_cache_path()){
			buffered_reader_.reset(new BufferedColorIMGAndAVIReader<Dtype>(
				fcn_img_data_reader_param.video_cache_path(),buffer_size_));
		}else{
			buffered_reader_.reset(new BufferedColorIMGAndAVIReader<Dtype>("",buffer_size_));
		}
		LOG(INFO)<<"In IImageBufferedDataReader, BufferedColorIMGAndAVIReader is used";
	}
	else
		buffered_reader_.reset(new BufferedColorJPGReader<Dtype>("",buffer_size_));
	if(this->random_rotate_degree_ != float(0)){
		LOG(ERROR)<<"Warning: random_rotate_degree is forced to be zero in ImagePair Data Reader";
	}
}



template <typename Dtype>
void IImageBufferedDataReader<Dtype>::SetResizeScale(int item_id, const Blob<Dtype> & img_pair_blob,
		Blob<Dtype> & dst_img_blob,const ImageDataSourceSampleType sample_type,
		const vector<Dtype>& coords,bool & is_neg,int scale_base_id)
{

	int lt_x,lt_y,rb_x,rb_y;
	this->GetResizeROIRange( lt_x,  lt_y,   rb_x,  rb_y,  item_id,
			img_pair_blob.width(),  img_pair_blob.height(),
			sample_type, coords, is_neg, scale_base_id);

	/**
	 * crop and resize.
	 */
//	CHECK_GE(rb_y  ,0);
//	CHECK_GE(lt_y,0);
//	CHECK_GE(rb_x ,0);
//	CHECK_GE(lt_x,0);
//	if(rb_y - lt_y < 0){
//		for(int i=0; i < coords.size()/2; ++i){
//			LOG(INFO)<<"point "<<i<<" = ["<<coords[i*2]<<", "<<coords[i*2+1];
//		}
//	}
	CHECK_GE(rb_y - lt_y,0);
	CHECK_GE(rb_x - lt_x,0);
//	LOG(INFO)<<"lt_x: "<<lt_x<<"  rb_x: "<<rb_x<<"  lt_y:"<<lt_y<<"  rb_y:"<<rb_y;
	dst_img_blob.Reshape(img_pair_blob.num(),img_pair_blob.channels(),rb_y - lt_y,rb_x - lt_x);

	if(this->use_gpu_){
		caffe::caffe_gpu_set(dst_img_blob.count(),Dtype(0),dst_img_blob.mutable_gpu_data());
		caffe::CropBlobs_gpu(img_pair_blob,0,lt_y,lt_x,rb_y,rb_x,dst_img_blob,0,0,0);
	}else{
		caffe::caffe_set(dst_img_blob.count(),Dtype(0),dst_img_blob.mutable_cpu_data());
		caffe::CropBlobs_cpu(img_pair_blob,0,lt_y,lt_x,rb_y,rb_x,dst_img_blob,0,0,0);
	}
	this->lt_x_[item_id] = lt_x;
	this->lt_y_[item_id] = lt_y;

}


template <typename Dtype>
void IImageBufferedDataReader<Dtype>::SetCropAndPad(int item_id,
		const Blob<Dtype> & src_img_blob,Blob<Dtype> & dst_img_blob,bool is_neg)
{
	/**
	 * crop Patch from resized image
	 */
	this->crop_begs_[item_id].first = 0;
	this->crop_begs_[item_id].second = 0;
	this->paddings_[item_id].first = 0;
	this->paddings_[item_id].second = 0;

	const int new_height = this->input_height_;
	const int new_width = this->input_width_;
	//Move the input ROI to the center of the image for Training, and randomly for Testing
	if (new_height < src_img_blob.height()) {

		if (this->cur_phase_ == TEST || (is_neg == false && this->cur_phase_ == TRAIN) )
			this->crop_begs_[item_id].second = (src_img_blob.height()/2 -new_height/2  );
		else
			this->crop_begs_[item_id].second = this->PrefetchRand() % (src_img_blob.height() - new_height + 1);

	} else {
		this->paddings_[item_id].second = (new_height - src_img_blob.height()) / 2;
	}

	if (new_width < src_img_blob.width()) {

		if (this->cur_phase_ == TEST || (is_neg == false && this->cur_phase_ == TRAIN) )
			this->crop_begs_[item_id].first = src_img_blob.width()/2 - new_width/2;
		else
			this->crop_begs_[item_id].first = this->PrefetchRand() % (src_img_blob.width()- new_width + 1);

	} else {
		this->paddings_[item_id].first = (new_width - src_img_blob.width()) / 2;
	}

	if(this->use_gpu_){
		caffe::caffe_gpu_set(dst_img_blob.count(),Dtype(0),dst_img_blob.mutable_gpu_data());
	}else{
		caffe::caffe_set(dst_img_blob.count(),Dtype(0),dst_img_blob.mutable_cpu_data());
	}

	int src_start_h = std::min(src_img_blob.height(),std::max(this->crop_begs_[item_id].second+this->rand_y_perturb_,0));
	int src_start_w = std::min(src_img_blob.width() ,std::max(this->crop_begs_[item_id].first+ this->rand_x_perturb_,0));

	int dst_start_h = this->paddings_[item_id].second;
	int dst_end_h   = new_height - this->paddings_[item_id].second;
	int dst_crop_h =  dst_end_h - dst_start_h;
	int dst_start_w = this->paddings_[item_id].first;
	int dst_end_w   = new_width - this->paddings_[item_id].first;
	int dst_crop_w =  dst_end_w - dst_start_w;

	int src_end_h = std::min(src_img_blob.height(), src_start_h + dst_crop_h);
	int src_end_w = std::min(src_img_blob.width(), src_start_w + dst_crop_w);
//	LOG(INFO)<<"this->crop_begs_[item_id].first:"<<this->crop_begs_[item_id].first
//			<<" this->crop_begs_[item_id].second:"<<this->crop_begs_[item_id].second;
//	LOG(INFO)<<"src_start_h:"<<src_start_h<<" src_start_w:"<<src_start_w
//			<<" src_end_h:"<<src_end_h<<" src_end_w:"<<src_end_w;
	if(this->use_gpu_){
		CropBlobs_gpu(src_img_blob,0,src_start_h,src_start_w,src_end_h,src_end_w,
				dst_img_blob,0,dst_start_h,dst_start_w);
	}else{
		CropBlobs_cpu(src_img_blob,0,src_start_h,src_start_w,src_end_h,src_end_w,
				dst_img_blob,0,dst_start_h,dst_start_w);
	}

}


template <typename Dtype>
void IImageBufferedDataReader<Dtype>::SetProcessParam(int item_id, Phase cur_phase){
	this->cur_phase_ = cur_phase;

	boost::unique_lock<boost::shared_mutex> write_lock(this->mutex_);

	while(item_id >= this->crop_begs_.size())
	{
		this->crop_begs_.push_back(pair<int, int>(0,0));
		this->paddings_.push_back(pair<int, int>(0,0));
		this->standard_scales_.push_back(1);
		this->random_scales_.push_back(1);
		this->sample_scales_.push_back(1);
		this->center_x_.push_back(0);
		this->center_y_.push_back(0);
		this->lt_x_.push_back(0);
		this->lt_y_.push_back(0);
		this->buff_blob_1_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
		this->buff_blob_2_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));

	}
	write_lock.unlock();

	this->rand_x_perturb_=0;
	this->rand_y_perturb_=0;
//	if(this->cur_phase_ == TRAIN)
//	{
//		this->rand_x_perturb_ = (int)(this->PrefetchRand()%(int)(this->heat_map_a_ *2))-(int)(this->heat_map_a_ );
//		this->rand_y_perturb_ = (int)(this->PrefetchRand()%(int)(this->heat_map_a_ *2))-(int)(this->heat_map_a_ );
//	}
}

template <typename Dtype>
bool IImageBufferedDataReader<Dtype>::ReadImgAndTransform(int item_id,Blob<Dtype>& prefetch_data,vector<cv::Mat*>  img_cv_ptrs,
		vector<cv::Mat*> img_ori_ptrs, pair< string, vector<Dtype> > & mutable_sample,
		vector<bool>& is_keypoint_transform_ignored, ImageDataSourceSampleType sample_type,
		Phase cur_phase,int scale_base_id  )
{
	/**
	 * resize crop_begs_, paddings_, standard_scales_, random_scales_ to
	 * fit the batch_size
	 */
	if(this->need_buffer_ == false){
		return IImageDataReader<Dtype>::ReadImgAndTransform(item_id,prefetch_data,img_cv_ptrs,img_ori_ptrs,
				mutable_sample,is_keypoint_transform_ignored,sample_type,cur_phase,scale_base_id);
	}

	cv::Mat & cv_img = *( (img_cv_ptrs[0]));
	cv::Mat & cv_img_depth = *( (img_cv_ptrs[1]));
	cv::Mat & cv_img_original = *( (img_ori_ptrs[0]));;
	cv::Mat & cv_img_original_depth = *( (img_ori_ptrs[1]));;

	this->SetProcessParam( item_id, cur_phase);


	vector<Dtype>& coords = mutable_sample.second;
	if(coords.size() == 0){
		sample_type = SOURCE_TYPE_ALL_NEGATIVE;
	}

	char img_path[512];


	/**
	 * Read Image
	 */
	Blob<Dtype>* ori_img_blob_ptr;
	Blob<Dtype>* temp_img_blob_ptr;
	if(this->single_thread_){
		ori_img_blob_ptr = (buff_blob_1_[0].get());
		temp_img_blob_ptr = (buff_blob_2_[0].get());
	}else{
		ori_img_blob_ptr = (buff_blob_1_[item_id].get());
		temp_img_blob_ptr = (buff_blob_2_[item_id].get());
	}
	Blob<Dtype>& ori_img_blob = *ori_img_blob_ptr;
	Blob<Dtype>& temp_img_blob = *temp_img_blob_ptr;


	if(is_video_img_){
		buffered_reader_->LoadToBlob(mutable_sample.first.c_str(),ori_img_blob);
	}else{
		sprintf(img_path, "%s.jpg", mutable_sample.first.c_str());
		buffered_reader_->LoadToBlob(img_path,ori_img_blob);
	}


//	LOG(INFO)<<"ori_img_blob.shape: "<<ori_img_blob.shape_string();
//	LOG(INFO)<<ori_img_blob.shape_string();
	if(is_img_pair_)
		CHECK_GE(ori_img_blob.channels(),6)<<"The input pair images should have at least 6 channels";
	else
		CHECK_EQ(ori_img_blob.channels(),3)<<"The input  image should have 3 channels";
	CHECK_EQ(ori_img_blob.num(),1)<<"The input pair image should have exact 1 num ";

	bool is_neg = true;
	SetResizeScale(item_id,ori_img_blob,temp_img_blob,sample_type,coords,is_neg,scale_base_id);


	if (this->pic_print_ || this->label_print_){
		LOG(INFO)<<"		sample_type:"<<sample_type <<"  standard_scales_:"<<this->standard_scales_[item_id]<<
		" random_scales_:"<<this->random_scales_[item_id]<<"  sample_scales_:"<<this->sample_scales_[item_id];
	}

	ori_img_blob.Reshape(temp_img_blob.num(),temp_img_blob.channels(),
			temp_img_blob.height() * this->sample_scales_[item_id],
			temp_img_blob.width() * this->sample_scales_[item_id]);

	if(this->use_gpu_){
		caffe::ResizeBlob_gpu(&temp_img_blob,&ori_img_blob);
	}else{
		caffe::ResizeBlob_cpu(&temp_img_blob,&ori_img_blob);
	}

	temp_img_blob.Reshape(ori_img_blob.num(),ori_img_blob.channels(),this->input_height_,this->input_width_);

	SetCropAndPad( item_id, ori_img_blob, temp_img_blob,is_neg);

	this->RefineCoords(item_id, coords,  is_keypoint_transform_ignored ,sample_type);

	if(this->random_rotate_degree_ != float(0)){
		LOG(ERROR)<<"Warning: random_rotate_degree is forced to be zero in ImagePair Data Reader";
	}


	if(this->use_gpu_){

		CUDA_CHECK(cudaMemcpy(prefetch_data.mutable_gpu_data() + prefetch_data.offset(item_id,0,0,0),
				temp_img_blob.gpu_data(), sizeof(Dtype) * prefetch_data.offset(1,0,0,0), cudaMemcpyDefault));

		ImgBlobRemoveMean_gpu( prefetch_data,
				this->mean_bgr_[0], this->mean_bgr_[1], this->mean_bgr_[2],0,item_id);
		if(is_img_pair_){
			ImgBlobRemoveMean_gpu( prefetch_data,
					this->mean2_bgr_[0], this->mean2_bgr_[1], this->mean2_bgr_[2],3,item_id);
		}
		cudaDeviceSynchronize();
	}else{

		memcpy(prefetch_data.mutable_cpu_data() + prefetch_data.offset(item_id,0,0,0) ,
				temp_img_blob.cpu_data(), sizeof(Dtype) * prefetch_data.offset(1,0,0,0));

		ImgBlobRemoveMean_cpu( prefetch_data,
				this->mean_bgr_[0], this->mean_bgr_[1], this->mean_bgr_[2],0,item_id);
		if(is_img_pair_){
			ImgBlobRemoveMean_cpu( prefetch_data,
					this->mean2_bgr_[0], this->mean2_bgr_[1], this->mean2_bgr_[2],3,item_id);
		}
	}
	if (this->pic_print_ || this->label_print_){
		LOG(INFO)<<"read to item_id:"<<item_id;

		cv_img = caffe::BlobImgDataToCVMat(temp_img_blob,0,Dtype(0),Dtype(0),Dtype(0),0);
//		temp_img_blob.scale_data(Dtype(255));
		if(is_img_pair_){
			cv_img_depth = caffe::BlobImgDataToCVMat(temp_img_blob,0,Dtype(0),Dtype(0),Dtype(0),3);
		}
		cv_img_original = caffe::BlobImgDataToCVMat(ori_img_blob,0,Dtype(0),Dtype(0),Dtype(0),0);
//		ori_img_blob.scale_data(Dtype(255));
		if(is_img_pair_){
			cv_img_original_depth = caffe::BlobImgDataToCVMat(ori_img_blob,0,Dtype(0),Dtype(0),Dtype(0),3);
		}
//		std::cout<<"data: "<<std::endl;
//		for(int i=0; i < 10; ++i){
//			std::cout<<prefetch_data.mutable_cpu_data()[i]/100<<" ";
//		}
//		std::cout<<std::endl;
	}

	return true;
}

INSTANTIATE_CLASS(IImageBufferedDataReader);
}  // namespace caffe

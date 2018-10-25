#include <string>
#include <vector>
#include <cmath>
#include "caffe/util/rng.hpp"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
IImageDataReader<Dtype>::IImageDataReader()
{
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	this->cur_phase_ = TRAIN;
	rand_x_perturb_ = 0;
	rand_y_perturb_ = 0;
}

template <typename Dtype>
IImageDataReader<Dtype>::~IImageDataReader()
{

}


template <typename Dtype>
void IImageDataReader<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_image_data_param);
	CHECK(fcn_image_data_param.has_fcn_image_data_reader_param())<<
			"fcn_image_data_reader_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_reader_param());
}

template <typename Dtype>
void IImageDataReader<Dtype>::SetUpParameter(
		const FCNImageDataReaderParameter& fcn_img_data_reader_param)
{

	this->scale_lower_limit_ = fcn_img_data_reader_param.scale_lower_limit();
	this->scale_upper_limit_ = fcn_img_data_reader_param.scale_upper_limit();

	CHECK_LE(this->scale_lower_limit_ , this->scale_upper_limit_);

	CHECK(fcn_img_data_reader_param.has_standard_len_point_1() &&
			fcn_img_data_reader_param.has_standard_len_point_2() &&
			fcn_img_data_reader_param.has_standard_len());
	this->standard_len_point_1_ = fcn_img_data_reader_param.standard_len_point_1();
	this->standard_len_point_2_ = fcn_img_data_reader_param.standard_len_point_2();
	this->standard_len_  = fcn_img_data_reader_param.standard_len();
	this->min_valid_standard_len_ = fcn_img_data_reader_param.min_valid_standard_len();
	this->restrict_roi_in_center_ = false;

	CHECK(fcn_img_data_reader_param.has_roi_center_point());
	this->roi_center_point_ = fcn_img_data_reader_param.roi_center_point();
	this->mean_bgr_[0] = fcn_img_data_reader_param.mean_b();
	this->mean_bgr_[1] = fcn_img_data_reader_param.mean_g();
	this->mean_bgr_[2] = fcn_img_data_reader_param.mean_r();
//	this->mean_d_ = fcn_img_data_reader_param.mean_d();
//	this->is_depth_img_ = fcn_img_data_reader_param.is_depth_img();
	this->random_rotate_degree_ = fcn_img_data_reader_param.random_rotate_degree();
	this->coord_jitter_ = fcn_img_data_reader_param.coord_jitter();
	this->random_roi_prob_ = fcn_img_data_reader_param.random_roi_prob();
	CHECK_GE(coord_jitter_,0);
	CHECK_GE(random_roi_prob_,0);
	CHECK_LE(random_roi_prob_,1);
	this->crop_begs_.clear();
	this->paddings_.clear();
	this->standard_scales_.clear();
	this->random_scales_.clear();
	this->sample_scales_.clear();
	this->center_x_.clear();
	this->center_y_.clear();
	this->lt_x_.clear();
	this->lt_y_.clear();

}


template <typename Dtype>
void IImageDataReader<Dtype>::GetResizeROIRange(int& lt_x, int& lt_y, int& rb_x, int& rb_y, int item_id,
		const int cv_mat_cols,  const int cv_mat_rows,
		const ImageDataSourceSampleType sample_type,
		const vector<Dtype>& coords,bool & is_neg,int scale_base_id){
	/**
	 * Get the scale for resize
	 */
	int ori_max_size = std::max(cv_mat_cols,cv_mat_rows);
	int in_max_size = std::max(this->input_height_,this->input_width_);

	CHECK_LT(scale_base_id, this->scale_base_.size());
	this->standard_scales_[item_id] = this->scale_base_[scale_base_id];
	center_x_[item_id] = MIN(this->input_width_ / 2,cv_mat_cols /2 );
	center_y_[item_id]  = MIN(this->input_height_ / 2,cv_mat_rows / 2);

	switch(sample_type)
	{
		case caffe::SOURCE_TYPE_POSITIVE_WITH_ROI:
		case  caffe::SOURCE_TYPE_POSITIVE:
		{
//			LOG(INFO)<<"								fuck!!!";
			this->random_scales_[item_id] = PrefetchRandFloat() * (scale_upper_limit_ - scale_lower_limit_)
					+ scale_lower_limit_;
			sample_scales_[item_id] = this->random_scales_[item_id] * this->standard_scales_[item_id];
			is_neg = true;
			if( standard_len_point_1_ * 2 +1  >= coords.size() || standard_len_point_2_ * 2 +1  >= coords.size() ||
					this->roi_center_point_ * 2 + 1 >= coords.size()  )
			{
				if(this->cur_phase_ == TRAIN  )
				{
					LOG(ERROR)<<"standard_len_point_1_ or standard_len_point_2_ or roi_center_point_ exceed the size of key point count";
				}
			}
			else
			{
				 //Handle situation when standard_len_points is available. We scale image according to
				 //standard_len
				if (coords[standard_len_point_1_ * 2] != -1 && coords[standard_len_point_2_ * 2] != -1 &&
					coords[standard_len_point_1_ * 2 +1] != -1 && coords[standard_len_point_2_ * 2+1] != -1)
				{
					const int len = standard_len_;
					float x_diff = fabs(coords[standard_len_point_2_ * 2] - coords[standard_len_point_1_ * 2]);
					float y_diff = fabs(coords[standard_len_point_2_ * 2 + 1] - coords[standard_len_point_1_ * 2 + 1]);
					float pt_diff = sqrtf(x_diff * x_diff + y_diff * y_diff);
					if(pt_diff >= min_valid_standard_len_ ){
						Dtype tmp_scale =   len / pt_diff  ;
						sample_scales_[item_id] *= tmp_scale;
						is_neg = false;
					}else {
						LOG(INFO)<<"Warning: Standard distance is too small. pt_diff = "<< pt_diff <<" , pt1 = ("<<
								coords[standard_len_point_1_ * 2]<<","<<coords[standard_len_point_1_ * 2 +1]<<") , pt2 = ("<<
								coords[standard_len_point_2_ * 2]<<","<<coords[standard_len_point_2_ * 2+1]<<").";
					}
				}
				//Handle situation when standard_len_points is not available. We randomly scale image.
				else
				{
					if(ori_max_size*scale_lower_limit_*this->standard_scales_[item_id] > in_max_size)
					{
						float min_scale_neg = in_max_size/(ori_max_size*scale_lower_limit_*this->standard_scales_[item_id]);
						float max_scale_neg = MAX( 2 /sample_scales_[item_id],min_scale_neg) ;
						Dtype temp_scale = PrefetchRandFloat() * (max_scale_neg - min_scale_neg) + min_scale_neg;
						sample_scales_[item_id] *= temp_scale;
					}
				}
				center_x_[item_id] =  coords[roi_center_point_ * 2] +
						standard_len_/sample_scales_[item_id]*coord_jitter_*(PrefetchRandFloat()*2-1);
				center_y_[item_id] =  coords[roi_center_point_ * 2 +1 ] +
						standard_len_/sample_scales_[item_id]*coord_jitter_*(PrefetchRandFloat()*2-1);
				// randomly set roi and scale with a probability
				if(PrefetchRandFloat() < random_roi_prob_){
					center_x_[item_id] =  MIN(cv_mat_cols ,
								MAX(0,int(PrefetchRandFloat() * cv_mat_cols  )));
					center_y_[item_id] =  MIN(cv_mat_rows ,
								MAX(0,int(PrefetchRandFloat() * cv_mat_rows  )));
					// (0.5 + (2-0.5)*PrefetchRandFloat()) means a rand(0.5,2)
					sample_scales_[item_id] *= (0.5 + (2-0.5)*PrefetchRandFloat());
				}
			}

			break;
		}
		case caffe::SOURCE_TYPE_ALL_NEGATIVE:
		{
			float min_scale_neg = in_max_size/(ori_max_size*this->standard_scales_[item_id]);
			float max_scale_neg = MAX( 2 ,min_scale_neg) ;
			this->random_scales_[item_id] = PrefetchRandFloat() * (max_scale_neg - min_scale_neg) + min_scale_neg;
			sample_scales_[item_id] = this->random_scales_[item_id] * this->standard_scales_[item_id];
			int patch_height = this->input_height_ / this->sample_scales_[item_id] +1;
			int patch_width = this->input_width_ / this->sample_scales_[item_id] +1;
			int min_x =  MIN(cv_mat_cols ,MAX(0,int( patch_width/2 +1 )));
			int min_y = MIN(cv_mat_rows ,MAX(0,int( patch_height/2 +1  )));
			int max_x =  MIN(cv_mat_cols ,MAX(0,int( cv_mat_cols - patch_width/2 -1 )));
			int max_y = MIN(cv_mat_rows ,MAX(0,int(  cv_mat_rows - patch_height/2 -1 )));
			center_x_[item_id] = PrefetchRandFloat() * (max_x - min_x) + min_x;
			center_y_[item_id] = PrefetchRandFloat() * (max_y - min_y) + min_y;
			break;
		}
		case caffe::SOURCE_TYPE_HARD_NEGATIVE:
		{
			/**
			 * get the	scale information , and crop the hard negative patch from image.
			 */
			CHECK_GE(coords.size(),4)<<"coords.size() for hard negativ should be larger than(or equal to ) 4.";
			this->random_scales_[item_id] = 1;
			sample_scales_[item_id] = this->random_scales_[item_id] * this->standard_scales_[item_id] * (this->input_height_+0.0)/(coords[3] - coords[1]);

			center_x_[item_id] =  (coords[0] + coords[2]) / 2;
			center_y_[item_id] =  (coords[1] + coords[3]) / 2;

			break;
		}
		default:
			LOG(FATAL) << "Unknown type " << sample_type;
	}
	/**
	 * crop ROI
	 */
	int patch_height = round(this->input_height_ / this->sample_scales_[item_id] );
	int patch_width = round(this->input_width_ / this->sample_scales_[item_id]);
	int lt_x_added = MAX(0, round(center_x_[item_id] + patch_width/2   ) - cv_mat_cols);
	lt_x =  MIN(cv_mat_cols , MAX(0,round(center_x_[item_id] - patch_width/2 )-lt_x_added));

	int lt_y_added = MAX(0, round(center_y_[item_id] + patch_height/2  ) - cv_mat_rows);
	lt_y = MIN(cv_mat_rows ,MAX(0,round(center_y_[item_id] - patch_height/2 )-lt_y_added ));

	int rb_x_added = MAX(0, 0 - round( center_x_[item_id] - patch_width/2 ));
	rb_x =  MIN(cv_mat_cols ,rb_x_added + MAX(0,round(center_x_[item_id] + patch_width/2 )));

	int rb_y_added = MAX(0, 0 - round(center_y_[item_id] - patch_height/2  ));
	rb_y = MIN(cv_mat_rows ,rb_y_added + MAX(0,round(center_y_[item_id] + patch_height/2  )));

	if(restrict_roi_in_center_){
		lt_x =   round(center_x_[item_id] - patch_width/2  );
		lt_y =round(center_y_[item_id] - patch_height/2 );
		rb_x =  round(center_x_[item_id] + patch_width/2  );
		rb_y = round( center_y_[item_id] + patch_height/2  );
	}

}

template <typename Dtype>
void IImageDataReader<Dtype>::SetResizeScale(int item_id, cv::Mat & cv_img_original_no_scaled,
		const ImageDataSourceSampleType sample_type,
		const vector<Dtype>& coords,bool & is_neg,int scale_base_id)
{
	int lt_x,lt_y,rb_x,rb_y;
	GetResizeROIRange( lt_x,  lt_y,   rb_x,  rb_y,  item_id,
			cv_img_original_no_scaled.cols,  cv_img_original_no_scaled.rows,
			 sample_type, coords, is_neg, scale_base_id);

	cv::Rect roi(lt_x, lt_y, rb_x - lt_x, rb_y - lt_y);
	cv::Mat cv_img_cropped_tmp;
	cv_img_original_no_scaled(roi).copyTo(cv_img_cropped_tmp);
	cv_img_cropped_tmp.copyTo(cv_img_original_no_scaled);
	lt_x_[item_id] = lt_x;
	lt_y_[item_id] = lt_y;
}

template <typename Dtype>
void IImageDataReader<Dtype>::SetCropAndPad(int item_id,const cv::Mat & cv_img_original,cv::Mat & cv_img,bool is_neg)
{
	/**
	 * crop Patch from resized image
	 */
	crop_begs_[item_id].first = 0;
	crop_begs_[item_id].second = 0;
	paddings_[item_id].first = 0;
	paddings_[item_id].second = 0;

	const int new_height = this->input_height_;
	const int new_width = this->input_width_;
	//Move the input ROI to the center of the image for Training, and randomly for Testing
	if (new_height < cv_img_original.rows) {

		if (this->cur_phase_ == TEST || (is_neg == false && this->cur_phase_ == TRAIN) )
			crop_begs_[item_id].second = (cv_img_original.rows/2 -new_height/2  );
		else
			crop_begs_[item_id].second = this->PrefetchRand() % (cv_img_original.rows - new_height + 1);

	} else {
	  paddings_[item_id].second = (new_height - cv_img_original.rows) / 2;

	}

	if (new_width < cv_img_original.cols) {

		if (this->cur_phase_ == TEST || (is_neg == false && this->cur_phase_ == TRAIN) )
			crop_begs_[item_id].first = cv_img_original.cols/2 - new_width/2;
		else
			crop_begs_[item_id].first = this->PrefetchRand() % (cv_img_original.cols - new_width + 1);

	} else {
	  paddings_[item_id].first = (new_width - cv_img_original.cols) / 2;

	}

	// padding zeros
	for (int h = 0; h < paddings_[item_id].second; ++h) {
	  for (int w = 0; w < new_width; ++w) {
		cv_img.at<cv::Vec3b>(h, w)[0] = 0;
		cv_img.at<cv::Vec3b>(h, w)[1] = 0;
		cv_img.at<cv::Vec3b>(h, w)[2] = 0;
	  }
	}
	for (int h = cv_img_original.rows + paddings_[item_id].second; h < new_height; ++h) {
	  for (int w = 0; w < new_width; ++w) {
		cv_img.at<cv::Vec3b>(h, w)[0] = 0;
		cv_img.at<cv::Vec3b>(h, w)[1] = 0;
		cv_img.at<cv::Vec3b>(h, w)[2] = 0;
	  }
	}
	for (int w = 0; w < paddings_[item_id].first; ++w) {
	  for (int h = paddings_[item_id].second; h < new_height - paddings_[item_id].second; ++h) {
		cv_img.at<cv::Vec3b>(h, w)[0] = 0;
		cv_img.at<cv::Vec3b>(h, w)[1] = 0;
		cv_img.at<cv::Vec3b>(h, w)[2] = 0;
	  }
	}
	for (int w = cv_img_original.cols + paddings_[item_id].first; w < new_width; ++w) {
	  for (int h = paddings_[item_id].second; h < new_height - paddings_[item_id].second; ++h) {
		cv_img.at<cv::Vec3b>(h, w)[0] = 0;
		cv_img.at<cv::Vec3b>(h, w)[1] = 0;
		cv_img.at<cv::Vec3b>(h, w)[2] = 0;
	  }
	}
	// copy original image data to input image
	for (int h = paddings_[item_id].second, y = crop_begs_[item_id].second;
		h < (new_height - paddings_[item_id].second); ++h, ++y)
	{
	  for (int w = paddings_[item_id].first, x = crop_begs_[item_id].first;
		  w < (new_width - paddings_[item_id].first); ++w, ++x)
	  {
		  int new_y = y + rand_y_perturb_;
		  int new_x = x + rand_x_perturb_;
		  if(new_y < cv_img_original.rows && new_x < cv_img_original.cols &&
				  new_x >=0 && new_y >= 0 )
		  {
			cv_img.at<cv::Vec3b>(h, w)[0] = cv_img_original.at<cv::Vec3b>(new_y, new_x)[0];
			cv_img.at<cv::Vec3b>(h, w)[1] = cv_img_original.at<cv::Vec3b>(new_y, new_x)[1];
			cv_img.at<cv::Vec3b>(h, w)[2] = cv_img_original.at<cv::Vec3b>(new_y, new_x)[2];
		  }
		  else
		  {
			cv_img.at<cv::Vec3b>(h, w)[0] = 0;
			cv_img.at<cv::Vec3b>(h, w)[1] = 0;
			cv_img.at<cv::Vec3b>(h, w)[2] = 0;
		  }
	  }
	}

}


template <typename Dtype>
void IImageDataReader<Dtype>::RefineCoords(int item_id,vector<Dtype>& coords,
		 vector<bool>& is_keypoint_ignored,const ImageDataSourceSampleType sample_type)
{
	// map coords to new positions
//		std::cout<<"before refine, coord:";
//		for(int i=0; i < coords.size(); ++i){
//			std::cout<<coords[i]<<" ";
//		}
//		std::cout<< std::endl;;


	switch(sample_type)
	{
		case caffe::SOURCE_TYPE_POSITIVE_WITH_ROI: {
			for(int i = 0; i < this->num_anno_points_per_instance_*2; ++i){
				coords[i] = -1;
			}
		}
		/* no break */
		case caffe::SOURCE_TYPE_POSITIVE : {
			CHECK_EQ(coords.size(), is_keypoint_ignored.size());
			CHECK_EQ(coords.size() % (this->num_anno_points_per_instance_*2), 0);

			for (int coords_i = 0; coords_i < coords.size(); coords_i += 2) {
				if(is_keypoint_ignored[coords_i] || is_keypoint_ignored[coords_i+1] )
					continue;
				if (coords[coords_i] != -1) {

					coords[coords_i] -= lt_x_[item_id];
					coords[coords_i + 1] -= lt_y_[item_id];

					coords[coords_i] *= sample_scales_[item_id];
					coords[coords_i + 1] *= sample_scales_[item_id];

					coords[coords_i] += paddings_[item_id].first;
					coords[coords_i + 1] += paddings_[item_id].second;

					coords[coords_i] -= crop_begs_[item_id].first;
					coords[coords_i + 1] -= crop_begs_[item_id].second;

					coords[coords_i] -=  rand_x_perturb_;
					coords[coords_i+1] -= rand_y_perturb_;
					}
				}
			break;
		}
		case caffe::SOURCE_TYPE_ALL_NEGATIVE: {
			coords.resize(this->num_anno_points_per_instance_*2, -1);
			is_keypoint_ignored.resize(this->num_anno_points_per_instance_*2, true);
			break;
		}
		case caffe::SOURCE_TYPE_HARD_NEGATIVE: {
			coords.resize(this->num_anno_points_per_instance_*2, -1);
			is_keypoint_ignored.resize(this->num_anno_points_per_instance_*2, true);
			break;
		}
		default:
			LOG(FATAL) << "Unknown type " << sample_type;
	}
//	std::cout<<"after refine, coord:";
//	for(int i=0; i < coords.size(); ++i){
//		std::cout<<coords[i]<<" ";
//	}
//	std::cout<< std::endl;;
}

template <typename Dtype>
void  IImageDataReader<Dtype>::Rotate(cv::Mat & cv_img,vector<Dtype>& coords,
		const vector<bool>& is_keypoint_ignored)
{
	const int new_height = this->input_height_;
	const int new_width = this->input_width_;

	cv::Mat temp_img = cv_img.clone();
	cv::Point2f center(new_width/2 +0.5,new_height/2+0.5);
	float degree = PrefetchRandFloat() * random_rotate_degree_ * 2 - random_rotate_degree_;

	cv::Mat M = cv::getRotationMatrix2D(center,degree,1);
	double m11,m12,m13,m21,m22,m23;
	m11 = M.at<double>(0,0);
	m12 = M.at<double>(0,1);
	m13 = M.at<double>(0,2);
	m21 = M.at<double>(1,0);
	m22 = M.at<double>(1,1);
	m23 = M.at<double>(1,2);

	cv::warpAffine(temp_img,cv_img,M,cv::Size(new_width,new_height), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS );
	// rotate annotations
	for(int temp_id = 0 ; temp_id < coords.size(); temp_id += 2)
	{
		if( is_keypoint_ignored[temp_id] || is_keypoint_ignored[temp_id+1] )
				  continue;
		if(coords[temp_id] == -1  || coords[temp_id+1] == -1)
					continue;
		coords[temp_id]  =  coords[temp_id] * m11 + coords[temp_id+1]*m12 + m13;
		coords[temp_id+1] = coords[temp_id] * m21 + coords[temp_id+1]*m22 + m23;
	}

}

template <typename Dtype>
void IImageDataReader<Dtype>::SetProcessParam(int item_id, Phase cur_phase){
	this->cur_phase_ = cur_phase;

	boost::unique_lock<boost::shared_mutex> write_lock(mutex_);

	while(item_id >= crop_begs_.size())
	{
		crop_begs_.push_back(pair<int, int>(0,0));
		paddings_.push_back(pair<int, int>(0,0));
		this->standard_scales_.push_back(1);
		this->random_scales_.push_back(1);
		sample_scales_.push_back(1);
		center_x_.push_back(0);
		center_y_.push_back(0);
		lt_x_.push_back(0);
		lt_y_.push_back(0);
	}
	write_lock.unlock();

	rand_x_perturb_=0;
	rand_y_perturb_=0;
//	if(this->cur_phase_ == TRAIN)
//	{
//		rand_x_perturb_ = (int)(this->PrefetchRand()%(int)(this->heat_map_a_ *2))-(int)(this->heat_map_a_ );
//		rand_y_perturb_ = (int)(this->PrefetchRand()%(int)(this->heat_map_a_ *2))-(int)(this->heat_map_a_ );
//	}
}
template <typename Dtype>
bool IImageDataReader<Dtype>::ReadImgAndTransform(int item_id,Blob<Dtype>& prefetch_data,vector<cv::Mat*>  img_cv_ptrs,
		vector<cv::Mat*> img_ori_ptrs,pair< string, vector<Dtype> > & mutable_sample,  vector<bool>& is_keypoint_ignored,
		ImageDataSourceSampleType sample_type,
		Phase cur_phase,int scale_base_id)
{
	/**
	 * resize crop_begs_, paddings_, standard_scales_, random_scales_ to
	 * fit the batch_size
	 */
	CHECK_GE(img_cv_ptrs.size(),1);
	CHECK_GE(img_ori_ptrs.size(),1);

	this->SetProcessParam( item_id, cur_phase);


	Dtype* top_data = prefetch_data.mutable_cpu_data();
	vector<Dtype>& coords = mutable_sample.second;
	if(coords.size() == 0){
		sample_type = SOURCE_TYPE_ALL_NEGATIVE;
	}

	char img_path[512];

	cv::Mat & cv_img = *( (img_cv_ptrs[0]));

	cv_img = cv::Mat(this->input_height_, this->input_width_, CV_8UC3);

	cv::Mat & cv_img_original = *( (img_ori_ptrs[0]));;

	/**
	 * Read Image
	 */
	cv::Mat cv_img_original_tmp;
	{
		sprintf(img_path, "%s.jpg", mutable_sample.first.c_str());
		cv_img_original_tmp = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
		if (!cv_img_original_tmp.data) {
		  sprintf(img_path, "%s.JPG",mutable_sample.first.c_str());
		  cv_img_original_tmp = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
		  if (!cv_img_original_tmp.data) {
			  LOG(ERROR)<< "Could not open or find file " << img_path;
			  return false;
		  }
		}
	}
	if (!cv_img_original_tmp.data) {
	  LOG(ERROR)<< "cv_img_original_tmp.data is empty!" ;
	  return false;
    }

	bool is_neg = true;
	SetResizeScale(item_id,cv_img_original_tmp,sample_type,coords,is_neg,scale_base_id);

	if (this->pic_print_ || this->label_print_)
		LOG(INFO)<<"		sample_type:"<<sample_type <<"  standard_scales_:"<<this->standard_scales_[item_id]<<" random_scales_:"<<
			this->random_scales_[item_id]<<"  sample_scales_:"<<sample_scales_[item_id];
	cv::resize(cv_img_original_tmp, cv_img_original,
			cv::Size(cv_img_original_tmp.cols * sample_scales_[item_id],
			cv_img_original_tmp.rows * sample_scales_[item_id]), 0, 0, cv::INTER_LINEAR);

	{
		SetCropAndPad( item_id, cv_img_original, cv_img,is_neg);
	}

	RefineCoords(item_id, coords,  is_keypoint_ignored,sample_type);
	if( random_rotate_degree_ != float(0))
	{
		Rotate( cv_img, coords, is_keypoint_ignored);
	}
	if (this->pic_print_ || this->label_print_)
		LOG(INFO)<<"read to item_id:"<<item_id;

	// copy input image data to blob
	{
		for (int c = 0; c < 3; ++c) {
		  for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
			  int top_index =  prefetch_data.offset(item_id, c, h, w);
			  CHECK_EQ(top_data[top_index], 0)<< "Duplicate Data Initializing...";
			  CHECK_LT(top_index, prefetch_data.count());

			  Dtype pixel = static_cast<Dtype>(cv_img.at<cv::Vec3b>(h, w)[c]);
			  top_data[top_index] = pixel - mean_bgr_[c];
			}
		  }
		}
	}

	return true;
}


template <typename Dtype>
void IImageDataReader<Dtype>:: PrintPic(int item_id, const string & output_path,
		cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type, const Dtype base_scale, const Blob<Dtype>& prefetch_data,
		string name_postfix)
{
	cv::Mat & cv_img = *( (img_cv_ptr));
	cv::Mat & cv_img_original = *( (img_ori_ptr));

	string out_sample_name = IImageDataProcessor<Dtype>::GetPrintSampleName(cur_sample,sample_type);
	LOG(INFO)<< "Saving sample: " << cur_sample.first;
	char path[512];
	{
		sprintf(path, "%s/pic/%03d_%s_s_base_scale_%f_input%s.jpg", output_path.c_str(), item_id,
				out_sample_name.c_str(), float(base_scale),name_postfix.c_str());
		imwrite(path, cv_img);
	}

	cv::Rect roi(crop_begs_[item_id].first, crop_begs_[item_id].second,
		this->input_width_ - paddings_[item_id].first, this->input_height_ - paddings_[item_id].second);
	{
		cv::rectangle(cv_img_original, roi, cv::Scalar(0, 0, 255), 5);
		sprintf(path, "%s/pic/%03d_%s_s_base_scale_%f_original%s.jpg", output_path.c_str(), item_id,
					out_sample_name.c_str(), float(base_scale),name_postfix.c_str());
		imwrite(path, cv_img_original);
	}
}

INSTANTIATE_CLASS(IImageDataReader);
}  // namespace caffe

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
BlockPacking<Dtype>::BlockPacking(){
	max_stride_ = 8;
	pad_h_  = pad_w_ = 0;
	max_block_size_ = 300;
	show_time_  = false;
	num_block_w_ = num_block_h_ = block_width_ = block_height_=0;
}

template <typename Dtype>
BlockPacking<Dtype>::~BlockPacking(){

}

template <typename Dtype>
void BlockPacking<Dtype>::SetUpParameter(const BlockPackingParameter& block_packing_param){
	max_stride_ = block_packing_param.max_stride();
	pad_h_ = block_packing_param.pad_h();
	pad_w_ = block_packing_param.pad_w();
	max_block_size_ = block_packing_param.max_block_size();

	pad_h_ = int( std::ceil(pad_h_/max_stride_))*max_stride_;
	pad_w_ = int( std::ceil(pad_w_/max_stride_))*max_stride_;
}


template <typename Dtype>
void BlockPacking<Dtype>::GetBlockingInfo1D(const int in_x, int & out_x, int& out_num){
	out_x = in_x;
	out_num = 1;
	if( out_x <= this->max_block_size_)
	{
		out_x = int( std::ceil(out_x/(max_stride_+0.0)))*max_stride_;
		return;
	}
	while(out_x >this->max_block_size_){
		out_num *= 2;
		out_x  = int( std::ceil(out_x/2.0));
	}

	out_x = int( std::ceil(out_x/(max_stride_+0.0)))*max_stride_;
}

/**
 *
 */
template <typename Dtype>
void BlockPacking<Dtype>::SetBlockPackingInfo(const Blob<Dtype>& blob_in){
	GetBlockingInfo1D(blob_in.height(), block_height_, num_block_h_);
	GetBlockingInfo1D(blob_in.width(), block_width_, num_block_w_);
}

/**
 * @brief Copy blob_in into buff_map_ for further processing.
 *
 */
template <typename Dtype>
void BlockPacking<Dtype>::SetInputBuff_cpu(const Blob<Dtype>& blob_in){
	int buff_map_h = block_height_ * num_block_h_;
	int buff_map_w = block_width_ * num_block_w_;
	buff_map_.Reshape(blob_in.num(),blob_in.channels(),buff_map_h, buff_map_w);
	caffe::caffe_set(buff_map_.count(),Dtype(0),buff_map_.mutable_cpu_data());
	for(int i= 0; i < blob_in.num(); ++i){
		caffe::CropBlobs_cpu(blob_in , i, 0, 0, blob_in.height(), blob_in.width(),
				buff_map_, i, 0, 0);
	}
}

/**
 *
 */
template <typename Dtype>
void BlockPacking<Dtype>::ImgPacking_cpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out){
	Timer timer;
	timer.Start();
	SetBlockPackingInfo(blob_in);
	SetInputBuff_cpu(blob_in);
	blob_out.Reshape(blob_in.num()*num_block_h_ * num_block_w_,blob_in.channels(),
			block_height_ + pad_h_*2, block_width_ + pad_w_*2);
	caffe::caffe_set(blob_out.count(),Dtype(0),blob_out.mutable_cpu_data());
        blob_out.mutable_gpu_data();
	for(int n=0; n < blob_in.num(); ++n){
		for(int h = 0; h < num_block_h_; ++h){
			for(int w = 0; w < num_block_w_; ++w){
				caffe::CropBlobs_cpu(buff_map_,n, h * block_height_ -pad_h_, w * block_width_ - pad_w_,
						h * block_height_ + pad_h_ + block_height_, w * block_width_ + pad_w_ + block_width_,
						blob_out, (num_block_h_ *n + h)* num_block_w_ +w,0,0);
			}
		}
	}
	if(show_time_)
		LOG(INFO)<<"Time for ImgPacking_cpu : "<<timer.MicroSeconds()/1000 << " milliseconds.";
}

/**
 *
 */
template <typename Dtype>
void BlockPacking<Dtype>::FeatureMapUnPacking_cpu(const Blob<Dtype>& blob_out, Blob<Dtype> blob_in,
					const int num_in_img ,int heat_map_a_){
	CHECK_EQ(num_in_img * num_block_h_ * num_block_w_ , blob_out.num());
	blob_in.Reshape(num_in_img,blob_out.channels(),blob_out.height() * num_block_h_,blob_out.width() * num_block_w_);
	caffe_set(blob_in.count(),Dtype(0),blob_in.mutable_cpu_data());
	int valid_heatmap_h = blob_out.height() - pad_h_*2/heat_map_a_ ;
	int valid_heatmap_w = blob_out.width() - pad_w_*2/heat_map_a_ ;
	for(int n=0; n < blob_in.num(); ++n){
		for(int h = 0; h < num_block_h_; ++h){
			for(int w = 0; w < num_block_w_; ++w){
				caffe::CropBlobs_cpu(blob_out, (num_block_h_ *n + h)* num_block_w_,
						pad_h_/heat_map_a_,pad_w_/heat_map_a_,
						blob_out.height() - pad_h_/heat_map_a_,blob_out.width() -pad_w_/heat_map_a_,
						blob_in,n, h * valid_heatmap_h , w * valid_heatmap_w );
			}
		}
	}
}



template <typename Dtype>
int BlockPacking<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start ){
	int param_size = 8;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = max_stride_;
	*(blob_cpu_data++) = pad_h_;
	*(blob_cpu_data++) = pad_w_;
	*(blob_cpu_data++) = max_block_size_;
	*(blob_cpu_data++) = num_block_w_;
	*(blob_cpu_data++) = num_block_h_;
	*(blob_cpu_data++) = block_width_;
	*(blob_cpu_data++) = block_height_;
	return start+param_size;
}

template <typename Dtype>
int BlockPacking<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start ){
	int param_size = 8;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	max_stride_ =  *(blob_cpu_data++);
	pad_h_ =  *(blob_cpu_data++);
	pad_w_ =  *(blob_cpu_data++);
	max_block_size_ =  *(blob_cpu_data++);
	num_block_w_ =  *(blob_cpu_data++);
	num_block_h_ =  *(blob_cpu_data++);
	block_width_ =  *(blob_cpu_data++);
	block_height_ =  *(blob_cpu_data++);
	return start+param_size;
}




// ############# RectBlockPacking #########

template <typename Dtype>
RoiRect<Dtype>::RoiRect(Dtype scale , Dtype start_y , Dtype start_x ,
		Dtype height , Dtype width ){
	scale_ = scale;
	start_y_ = start_y;
	start_x_ = start_x;
	height_ = height;
	width_ = width;
}

template <typename Dtype>
int RoiRect<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start ){
	int param_size = 5;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = scale_;
	*(blob_cpu_data++) = start_y_;
	*(blob_cpu_data++) = start_x_;
	*(blob_cpu_data++) = height_;
	*(blob_cpu_data++) = width_;
	return start+param_size;
}
template <typename Dtype>
int RoiRect<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start  ){
	int param_size = 5;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	scale_ = *(blob_cpu_data++);
	start_y_ = *(blob_cpu_data++);
	start_x_ = *(blob_cpu_data++);
	height_ = *(blob_cpu_data++);
	width_ = *(blob_cpu_data++);
	return start+param_size;
}

template <typename Dtype>
void RectBlockPacking<Dtype>::setRoi(const Blob<Dtype>& blob_in,
		const vector<Dtype> scales,const int max_size){

	int max_input_size = max(blob_in.height() , blob_in.width());
	Dtype acture_scale = min(Dtype(1), max_size/Dtype(max_input_size ));


	roi_.clear();
	for(int i=0 ; i < scales.size(); ++i){
		roi_.push_back(RoiRect<Dtype>(scales[i]*acture_scale,0,0, blob_in.height() , blob_in.width() ));
	}
	sort(roi_.begin(), roi_.end(), RoiRect<Dtype>::greaterScaledArea);
}


template <typename Dtype>
void RectBlockPacking<Dtype>::setRoi(const Blob<Dtype>& blob_in,
		const pair<string, vector<Dtype> >&  cur_sample){
	const vector<Dtype>& roi_info = cur_sample.second;
	int n_roi = roi_info.size()/5;
	const int in_height = blob_in.height();
	const int in_width = blob_in.width();
	CHECK_EQ(roi_info.size()%5,0);
	CHECK_GT(n_roi,0);
	roi_.clear();
	for(int i=0; i<n_roi; ++i){
		roi_.push_back(RoiRect<Dtype>(pow(2, roi_info[i*5+4]),int(roi_info[i*5+1]),int(roi_info[i*5+0]),
				MIN(in_height,int(roi_info[i*5+3] - roi_info[i*5+1])) ,
				MIN(in_width,int(roi_info[i*5+2] - roi_info[i*5+0] ))));
//		RoiRect<Dtype> t = RoiRect<Dtype>(pow(2, roi_info[i*5+4]),roi_info[i*5+1],roi_info[i*5+0],
//				roi_info[i*5+3] - roi_info[i*5+1]  , roi_info[i*5+2] - roi_info[i*5+0] );
//		LOG(INFO)<<"roi: "<< t;
	}
	sort(roi_.begin(), roi_.end(), RoiRect<Dtype>::greaterMaxScaledEdge);
}


template <typename Dtype>
void RectBlockPacking<Dtype>::SetBlockPackingInfo(const Blob<Dtype>& blob_in){
	if(roi_.empty()){
		setRoi(blob_in,vector<Dtype>(1,1));
	}
	rectMap_.Clear();
	for(int i = 0 ; i < roi_.size(); ++i){
		Rect rect = Rect(RectPoint(), int(roi_[i].GetScaledHeight() + this->pad_h_),
				int(roi_[i].GetScaledWidth() + this->pad_w_));
		rectMap_.PlaceRect(rect);
	}
	placedRect_ = rectMap_.GetPlacedRects();
	this->GetBlockingInfo1D(rectMap_.MapHeight() - this->pad_h_, this->block_height_, this->num_block_h_);
	this->GetBlockingInfo1D(rectMap_.MapWidth() - this->pad_w_, this->block_width_, this->num_block_w_);
}

/**
 *
 * @brief Copy blob_in into buff_map_ for further processing.
 * 		Only support when blob_in.num() == 1 .
 */
template <typename Dtype>
void RectBlockPacking<Dtype>::SetInputBuff_cpu(const Blob<Dtype>& blob_in){

	CHECK_EQ(blob_in.num(),1);
	Timer timer, timer_crop_1, timer_crop_2, timer_resize, reshape;
	Dtype time_crop_1 = 0;
	Dtype time_crop_2 = 0;
	Dtype time_resize = 0;
	timer.Start();
	int buff_map_h = this->block_height_ * this->num_block_h_;
	int buff_map_w = this->block_width_ * this->num_block_w_;
	this->buff_map_.Reshape(blob_in.num(),blob_in.channels(),buff_map_h, buff_map_w);
	caffe::caffe_set(this->buff_map_.count(),Dtype(0),this->buff_map_.mutable_cpu_data());
	const vector<Rect>& rects = rectMap_.GetPlacedRects();
	CHECK_EQ(roi_.size(), rects.size());
	/**
	 * crop and resize
	 */
	for(int i= 0; i < rects.size(); ++i){
		buff_blob_1_.Reshape(1,blob_in.channels(),roi_[i].height_,roi_[i].width_ );
		buff_blob_2_.Reshape(1,blob_in.channels(),int(roi_[i].GetScaledHeight() ),
				int(roi_[i].GetScaledWidth() ));
		CHECK_EQ(rects[i].height,buff_blob_2_.height() + this->pad_h_);
		CHECK_EQ(rects[i].width ,buff_blob_2_.width()+ this->pad_w_);
		timer_crop_1.Start();
		caffe::CropBlobs_cpu(blob_in,roi_[i].start_y_,roi_[i].start_x_,
				roi_[i].start_y_ + roi_[i].height_,roi_[i].start_x_+ roi_[i].width_,buff_blob_1_);
		time_crop_1 += timer_crop_1.MicroSeconds();
		timer_resize.Start();
		caffe::ResizeBlob_cpu(&buff_blob_1_,&buff_blob_2_);
		time_resize += timer_resize.MicroSeconds();
		timer_crop_2.Start();
		caffe::CropBlobs_cpu(buff_blob_2_ , 0, 0, 0, buff_blob_2_.height(), buff_blob_2_.width(),
				this->buff_map_, 0, rects[i].left_top.y, rects[i].left_top.x);
		time_crop_2 += timer_crop_2.MicroSeconds();
	}
	if(this->show_time_)
		LOG(INFO)<<"Time for PatchWork(in ImgPacking): "<<timer.MicroSeconds()/1000<<
			" milliseconds,in which crop_1 takes "<< time_crop_1/1000<<
			" , resize takes: "<<time_resize/1000 <<"  and crop_2 takes "<<time_crop_2/1000;
}


/**
 * Serialization for Rect
 */
template <typename Dtype>
int RectBlockPacking<Dtype>::SerializeToBlob(Blob<Dtype>& blob, Rect& rect, int start ){
	int param_size = 4;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = rect.left_top.y;
	*(blob_cpu_data++) = rect.left_top.x;
	*(blob_cpu_data++) = rect.height;
	*(blob_cpu_data++) = rect.width;
	return start+param_size;
}

template <typename Dtype>
int RectBlockPacking<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, Rect& rect, int start  ){
	int param_size = 4;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	rect.left_top.y = *(blob_cpu_data++) ;
	rect.left_top.x = *(blob_cpu_data++) ;
	rect.height = *(blob_cpu_data++) ;
	rect.width = *(blob_cpu_data++) ;
	return start+param_size;
}
/**
 * rule: first save BlobPacking param, then RoiRect and RectMap
 */
template <typename Dtype>
int RectBlockPacking<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start  ){
	int cur_start = BlockPacking<Dtype>::SerializeToBlob(blob,start);
	Dtype* blob_cpu_data = blob.mutable_cpu_data();
	*(blob_cpu_data + cur_start++) = roi_.size();
	for(int i=0 ; i < roi_.size();++i){
		cur_start = roi_[i].SerializeToBlob(blob, cur_start);
	}
	*(blob_cpu_data + cur_start++) = placedRect_.size();
	for(int i=0; i <placedRect_.size(); ++i ){
		cur_start = SerializeToBlob(blob, placedRect_[i], cur_start);
	}
	return cur_start;
}

template <typename Dtype>
int RectBlockPacking<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start  ){
	int cur_start = BlockPacking<Dtype>::ReadFromSerialized(blob,start);
	const Dtype* blob_cpu_data = blob.cpu_data();
	int roi_size = *(blob_cpu_data + cur_start++);
	roi_.clear();
	for(int i=0 ; i < roi_size;++i){
		roi_.push_back(RoiRect<Dtype>());
		cur_start = roi_[i].ReadFromSerialized(blob, cur_start);
	}
	int rect_size = *(blob_cpu_data + cur_start++);
	placedRect_.clear();
	for(int i=0; i <rect_size; ++i ){
		placedRect_.push_back(Rect());
		cur_start = ReadFromSerialized(blob, placedRect_[i], cur_start);
	}
	return cur_start;
}


template <typename Dtype>
int RectBlockPacking<Dtype>::GetRoiIdByBufferedImgCoords(const int coords_y,
		const int coords_x){
	RectPoint point(coords_y, coords_x);
	RectPoint point2(coords_y + this->pad_h_, coords_x + this->pad_w_);
	for(int i=0; i < placedRect_.size(); ++i){
		if(placedRect_[i].Contain(point) && placedRect_[i].Contain(point2)){
			return i;
		}
	}
	return -1;
}

// ############# PyramidImageDataLayer #########
template <typename Dtype>
PyramidImageDataLayer<Dtype>::~PyramidImageDataLayer<Dtype>() {
	JoinPrefetchThread();
}

/**
 *
 */
template <typename Dtype>
void PyramidImageDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	CHECK(this->layer_param_.has_pyramid_data_param());


	scale_start_ = this->layer_param_.pyramid_data_param().scale_start();
	scale_end_ = this->layer_param_.pyramid_data_param().scale_end();
	scale_step_ = this->layer_param_.pyramid_data_param().scale_step();
	scale_from_annotation_ = this->layer_param_.pyramid_data_param().scale_from_annotation();
	CHECK_GE(scale_end_ , scale_start_);
	CHECK_GT(scale_step_,0);

	heat_map_a_ = this->layer_param_.pyramid_data_param().heat_map_a();
	heat_map_b_ = this->layer_param_.pyramid_data_param().heat_map_b();
	this->mean_bgr_[0] = this->layer_param_.pyramid_data_param().mean_b();
	this->mean_bgr_[1] = this->layer_param_.pyramid_data_param().mean_g();
	this->mean_bgr_[2] = this->layer_param_.pyramid_data_param().mean_r();

	this->mean2_bgr_[0] = this->layer_param_.pyramid_data_param().mean2_b();
	this->mean2_bgr_[1] = this->layer_param_.pyramid_data_param().mean2_g();
	this->mean2_bgr_[2] = this->layer_param_.pyramid_data_param().mean2_r();
	this->is_img_pair_ = this->layer_param_.pyramid_data_param().is_img_pair();

	max_block_num_ = this->layer_param_.pyramid_data_param().max_block_num();

	max_input_size_ = this->layer_param_.pyramid_data_param().max_input_size();

	cur_sample_1_.second.clear();
	cur_sample_2_.second.clear();
	cur_sample_list_.clear();
	cur_sample_list_.push_back(&cur_sample_1_);
	cur_sample_list_.push_back(&cur_sample_2_);
	img_w_[0] = img_w_[1] = img_h_[0] = img_h_[1] = 0;
	rect_block_packer_list_.clear();
	rect_block_packer_list_.push_back(&rect_block_packer_1_);
	rect_block_packer_list_.push_back(&rect_block_packer_2_);

	img_blob_list_.clear();

	img_blob_list_.push_back(&img_blob_1_);
	img_blob_list_.push_back(&img_blob_2_);

	buffered_block_.clear();
	buffered_block_.push_back(&buffered_block_1_);
	buffered_block_.push_back(&buffered_block_2_);
	used_buffered_block_id_ = buffered_block_.size()-1;

	CHECK(this->layer_param_.pyramid_data_param().has_block_packing_param());
	rect_block_packer_1_.SetUpParameter(this->layer_param_.pyramid_data_param().block_packing_param());
	rect_block_packer_2_.SetUpParameter(this->layer_param_.pyramid_data_param().block_packing_param());



	if(this->img_from_memory_ == false){
		shuffle_ = this->layer_param_.pyramid_data_param().shuffle();
		CHECK(this->layer_param_.pyramid_data_param().has_image_list_file());
		CHECK(this->layer_param_.pyramid_data_param().has_image_folder());
		samples_provider_.SetShuffleFlag(shuffle_);
		samples_provider_.ReadSamplesFromFile(this->layer_param_.pyramid_data_param().image_list_file(),
				this->layer_param_.pyramid_data_param().image_folder(),5,true);
		CreatePrefetchThread();
	}


	forward_iter_id_ = 0 ;

	cur_sample_id_ = -1;
	int max_stride_ = rect_block_packer_1_.max_stride();
	int max_block_size_ = 10 *max_stride_; //rect_block_packer_1_.max_block_size();
	int out_size = int( std::ceil(max_block_size_/max_stride_))*max_stride_;
	if (this->pic_print_)
		LOG(INFO)<<"max_stride_: "<<max_stride_<<" max_block_size_: "<<max_block_size_ <<
			"  out_size: "<<out_size;
	if(this->is_img_pair_){
		top[0]->Reshape(max_block_num_,6,out_size, out_size);
	}else{
		top[0]->Reshape(max_block_num_,3,out_size, out_size);
	}
	top[1]->Reshape(1,1,50,1024);
	this->mode_ = Caffe::mode();


	show_output_path_ = string("cache/PyramidDataLayer");
	pic_print_ = ((getenv("PIC_PRINT") != NULL) && (getenv("PIC_PRINT")[0] == '1'));
	char output_path[512];
	if (this->pic_print_) {
		sprintf(output_path, "%s/pic",this->show_output_path_.c_str());
		CreateDir(output_path);
	}
	show_time_ = ((getenv("SHOW_TIME") != NULL) && (getenv("SHOW_TIME")[0] == '1'));
}

template <typename Dtype>
void PyramidImageDataLayer<Dtype>::CreatePrefetchThread(){
	CHECK(StartInternalThread()) << "Thread in PyramidImageDataLayer execution failed";
}

template <typename Dtype>
void PyramidImageDataLayer<Dtype>::JoinPrefetchThread(){
	CHECK(WaitForInternalThreadToExit()) << "Thread in PyramidImageDataLayer joining failed";
}




template <typename Dtype>
void PyramidImageDataLayer<Dtype>::LoadSampleToImgBlob(
		pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_){
	cur_sample_ = samples_provider_.GetOneSample() ;
	int cv_read_flag = CV_LOAD_IMAGE_COLOR;
	if(this->is_img_pair_){
		string cur_name = cur_sample_.first+string(".jpg");
		cv::Mat cv_img = cv::imread(cur_name, cv_read_flag);
		if (!cv_img.data) {
			LOG(ERROR) << "Could not open or find file " << cur_name;
		}
		cur_name = cur_sample_.first+string("_depth.jpg");
		cv::Mat cv_img2 = cv::imread(cur_name, cv_read_flag);
		if (!cv_img2.data) {
			LOG(ERROR) << "Could not open or find file " << cur_name;
		}
		ReadImgToBlob(  cv_img,cv_img2, img_blob_);

		ImgBlobRemoveMean(img_blob_, mean_bgr_[0],mean_bgr_[1],mean_bgr_[2],0);
	}else{
		if(caffe::ReadImgToBlob(cur_sample_.first+string(".jpg"),img_blob_,
						 mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]) ==  false){
                  if(caffe::ReadImgToBlob(cur_sample_.first+string(".png"),img_blob_,
                                                 mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]) ==  false){
			CHECK(caffe::ReadImgToBlob(cur_sample_.first +string(".JPG"),img_blob_,
					mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]))<<
						"cannot find image  : "<<cur_sample_.first;}
		}
	}
}


template <typename Dtype>
void PyramidImageDataLayer<Dtype>::SetRoiAndScale(RectBlockPacking<Dtype>& rect_block_packer_,
		pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_){
	if(scale_from_annotation_ == false){
		vector<Dtype> scales ;
		scales.clear();
		int num_scale = (scale_end_ - scale_start_)/scale_step_;
		if(num_scale > 30)
			LOG(INFO)<<"Warning, too much testing scales: "<< num_scale;
		for(Dtype scale = scale_start_; scale <= scale_end_; scale += scale_step_){
			Dtype cal_scale = pow(2,scale);
			scales.push_back(cal_scale);
		}
		if (this->pic_print_)
			LOG(INFO)<<"img_blob height:"<<img_blob_.height()<<"  width: "<<img_blob_.width();
		rect_block_packer_.setRoi(img_blob_, scales,max_input_size_);
	}else{
		rect_block_packer_.setRoi(img_blob_, cur_sample_);
	}
}

/**
 *
 */
template <typename Dtype>
void PyramidImageDataLayer<Dtype>::InternalThreadEntry(){
//	this->mode_ = Caffe::mode();
	Timer prefetch_timer, read_timer;
	prefetch_timer.Start();
	read_timer.Start();
	int buff_block_id = (used_buffered_block_id_+1)%(buffered_block_.size());
	Blob<Dtype>& buff_block = *(buffered_block_[buff_block_id]);
	pair<string, vector<Dtype> >&  cur_sample_ = *(cur_sample_list_[buff_block_id]);
	Blob<Dtype>& img_blob_ = *(img_blob_list_[buff_block_id]);
	RectBlockPacking<Dtype>& rect_block_packer_ =  *(rect_block_packer_list_[buff_block_id]);
	rect_block_packer_.setShowTime(show_time_);


	LoadSampleToImgBlob(cur_sample_, img_blob_);


	if(show_time_){
		LOG(INFO)<< "Time for reading image in pyramid_data_layer prefetch: "
		   <<read_timer.MilliSeconds()<< " milliseconds.";
	}
	img_w_[buff_block_id] = img_blob_.width();
	img_h_[buff_block_id] = img_blob_.height();
	/**
	 * set scales for patchwork.
	 */
	SetRoiAndScale( rect_block_packer_, cur_sample_,  img_blob_);


	if(this->mode_ == Caffe::GPU)
		rect_block_packer_.ImgPacking_gpu(img_blob_, buff_block);
	else
		rect_block_packer_.ImgPacking_cpu(img_blob_, buff_block);
	if(this->mode_ == Caffe::GPU){
		Timer synchronize_timer;
		synchronize_timer.Start();
		buff_block.mutable_gpu_data();
		if(show_time_){
			LOG(INFO)<< "Time for synchronizing buff memory in prefetch: "<<
					synchronize_timer.MilliSeconds()<<" milliseconds.";
		}
	}
	if(show_time_){
		LOG(INFO)<< "Time for pyramid_data_layer prefetch: "<<
			prefetch_timer.MilliSeconds()<<" milliseconds.";
	}
}


template <typename Dtype>
void PyramidImageDataLayer<Dtype>::ShowImg(const vector<Blob<Dtype>*>& top){
	char path[512];
	pair<string, vector<Dtype> >&  cur_sample_ = *(cur_sample_list_[used_buffered_block_id_]);
	string img_name = ImageDataSourceProvider<Dtype>::GetSampleName(cur_sample_);

	RectBlockPacking<Dtype>& rect_block_packer_ =  *(rect_block_packer_list_[used_buffered_block_id_]);

	cv::Mat packed_img = BlobImgDataToCVMat(rect_block_packer_.buff_map(),0,
			 mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]);

	sprintf(path, "%s/pic/%s_packed_img.jpg", show_output_path_.c_str(),img_name.c_str());
	LOG(INFO)<<"saving "<<path;
	imwrite(path,packed_img);

	for(int i=0; i < top[0]->num();++i){
		cv::Mat packed_block = BlobImgDataToCVMat(*(top[0]),i,
			mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]);
		sprintf(path, "%s/pic/%s_packed_block_%d.jpg", show_output_path_.c_str(),img_name.c_str(),i+forward_iter_id_ * max_block_num_);
		LOG(INFO)<<"saving "<<path;
		imwrite(path,packed_block);
	}

}

template <typename Dtype>
void PyramidImageDataLayer<Dtype>::StartProcessOneImg(){
	JoinPrefetchThread();
	used_buffered_block_id_ = (used_buffered_block_id_+1)%(buffered_block_.size());
	forward_times_for_cur_sample_ = std::ceil((buffered_block_[used_buffered_block_id_])->num()/(max_block_num_+0.0));
	cur_sample_id_ = (cur_sample_id_+1)/GetTotalSampleSize();
	CreatePrefetchThread();
}

/**
 * Forward: Copy the buffered blocks into top[0], and RectBlockPacking parameters to
 * top[1]
 */
template <typename Dtype>
void PyramidImageDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Timer total, prefetch;
	total.Start();
	prefetch.Start();
	if(forward_iter_id_ == 0 ){
		StartProcessOneImg();
	}
	Dtype prefetch_time = prefetch.MicroSeconds();
	Blob<Dtype>& img_blob_ = *(img_blob_list_[used_buffered_block_id_]);
	Blob<Dtype>& buff_block = *(buffered_block_[used_buffered_block_id_]);

	int start_id = forward_iter_id_ * max_block_num_;
	int end_id = MIN(forward_iter_id_ * max_block_num_ +max_block_num_ ,buff_block.num());

	top[0]->Reshape(end_id - start_id,img_blob_.channels(),
			buff_block.height(), buff_block.width());
	int block_length = img_blob_.channels() *buff_block.height()*buff_block.width();
	for(int i= start_id; i < end_id; ++i){
		caffe::caffe_copy(block_length,buff_block.cpu_data()+buff_block.offset(i,0),
				top[0]->mutable_cpu_data()+top[0]->offset(i-start_id,0));
	}
	caffe_set(top[1]->count(),Dtype(0),top[1]->mutable_cpu_data());
	SerializeToBlob(*(top[1]), 0);

	if (this->pic_print_)
		this->ShowImg(top);
	forward_iter_id_ =  (forward_iter_id_+1) % forward_times_for_cur_sample_;
	if(show_time_){
		LOG(INFO)<<"Time for PyramidImageDataLayer::Forward_cpu:  "<<total.MicroSeconds()/1000.0<<
			" milliseconds, in which prefetch cost "<< prefetch_time/1000;
	}
}

// ############# PyramidImageDataParam #########

template <typename Dtype>
int PyramidImageDataParam<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start ){
	int param_size = 7;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	img_w_ = *(blob_cpu_data++) ;
	img_h_ = *(blob_cpu_data++) ;
	heat_map_a_ = *(blob_cpu_data++) ;
	heat_map_b_ = *(blob_cpu_data++) ;
	max_block_num_ = *(blob_cpu_data++) ;
	forward_times_for_cur_sample_ = *(blob_cpu_data++) ;
	forward_iter_id_ = *(blob_cpu_data++) ;
	return rect_block_packer_.ReadFromSerialized(blob,start+param_size);
}

template <typename Dtype>
int PyramidImageDataLayer<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start ){
	int param_size = 7;
	CHECK_GE(blob.count(), start+param_size);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = img_w_[used_buffered_block_id_];
	*(blob_cpu_data++) = img_h_[used_buffered_block_id_];
	*(blob_cpu_data++) = heat_map_a_;
	*(blob_cpu_data++) = heat_map_b_;
	*(blob_cpu_data++) = max_block_num_;
	*(blob_cpu_data++) = forward_times_for_cur_sample_;
	*(blob_cpu_data++) = forward_iter_id_;
	RectBlockPacking<Dtype>& rect_block_packer_ =  *(rect_block_packer_list_[used_buffered_block_id_]);
	return rect_block_packer_.SerializeToBlob(blob,start+param_size);

}

/**
 *
 */
template <typename Dtype>
void BlockPacking<Dtype>::ImgPacking_gpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out){
	Timer timer;
	timer.Start();
	SetBlockPackingInfo(blob_in);
	SetInputBuff_gpu(blob_in);
	blob_out.Reshape(blob_in.num()*num_block_h_ * num_block_w_,blob_in.channels(),
			block_height_ + pad_h_*2, block_width_ + pad_w_*2);
	caffe::caffe_gpu_set(blob_out.count(),Dtype(0),blob_out.mutable_gpu_data());
	for(int n=0; n < blob_in.num(); ++n){
		for(int h = 0; h < num_block_h_; ++h){
			for(int w = 0; w < num_block_w_; ++w){
				caffe::CropBlobs_gpu(buff_map_,n, h * block_height_ -pad_h_, w * block_width_ - pad_w_,
						h * block_height_ + pad_h_ + block_height_, w * block_width_ + pad_w_ + block_width_,
						blob_out, (num_block_h_ *n + h)* num_block_w_ +w,0,0);
			}
		}
	}
	if(show_time_)
		LOG(INFO)<<"Time for ImgPacking_gpu : "<<timer.MicroSeconds()/1000 << " milliseconds.";
}


template <typename Dtype>
void BlockPacking<Dtype>::FeatureMapUnPacking_gpu(const Blob<Dtype>& blob_out, Blob<Dtype> blob_in,
					const int num_in_img ,int heat_map_a_){
	CHECK_EQ(num_in_img * num_block_h_ * num_block_w_ , blob_out.num());
	blob_in.Reshape(num_in_img,blob_out.channels(),blob_out.height() * num_block_h_,blob_out.width() * num_block_w_);
	caffe_gpu_set(blob_in.count(),Dtype(0),blob_in.mutable_gpu_data());
	int valid_heatmap_h = blob_out.height() - pad_h_*2/heat_map_a_ ;
	int valid_heatmap_w = blob_out.width() - pad_w_*2/heat_map_a_ ;
	for(int n=0; n < blob_in.num(); ++n){
		for(int h = 0; h < num_block_h_; ++h){
			for(int w = 0; w < num_block_w_; ++w){
				caffe::CropBlobs_gpu(blob_out, (num_block_h_ *n + h)* num_block_w_,
						pad_h_/heat_map_a_,pad_w_/heat_map_a_,
						blob_out.height() - pad_h_/heat_map_a_,blob_out.width() -pad_w_/heat_map_a_,
						blob_in,n, h * valid_heatmap_h , w * valid_heatmap_w );
			}
		}
	}
}

/**
 *
 * @brief Copy blob_in into buff_map_ for further processing.
 * 		Only support when blob_in.num() == 1 .
 */
template <typename Dtype>
void RectBlockPacking<Dtype>::SetInputBuff_gpu(const Blob<Dtype>& blob_in){

	CHECK_EQ(blob_in.num(),1);
	Timer timer, timer_crop_1, timer_crop_2, timer_resize, reshape;
	Dtype time_crop_1 = 0;
	Dtype time_crop_2 = 0;
	Dtype time_resize = 0;
	timer.Start();
	int buff_map_h = this->block_height_ * this->num_block_h_;
	int buff_map_w = this->block_width_ * this->num_block_w_;
	this->buff_map_.Reshape(blob_in.num(),blob_in.channels(),buff_map_h, buff_map_w);
	caffe::caffe_gpu_set(this->buff_map_.count(),Dtype(0),this->buff_map_.mutable_gpu_data());
	const vector<Rect>& rects = rectMap_.GetPlacedRects();
	CHECK_EQ(roi_.size(), rects.size());
	/**
	 * crop and resize
	 */
	for(int i= 0; i < rects.size(); ++i){
		buff_blob_1_.Reshape(1,blob_in.channels(),roi_[i].height_,roi_[i].width_ );
		buff_blob_2_.Reshape(1,blob_in.channels(),int(roi_[i].GetScaledHeight() ),
				int(roi_[i].GetScaledWidth() ));
//		CHECK_EQ(rects[i].height,buff_blob_2_.height() + this->pad_h_);
//		CHECK_EQ(rects[i].width ,buff_blob_2_.width()+ this->pad_w_);
		timer_crop_1.Start();
		caffe::CropBlobs_gpu(blob_in,roi_[i].start_y_,roi_[i].start_x_,
				roi_[i].start_y_ + roi_[i].height_,roi_[i].start_x_+ roi_[i].width_,buff_blob_1_);
		time_crop_1 += timer_crop_1.MicroSeconds();
		timer_resize.Start();
		caffe::ResizeBlob_gpu(&buff_blob_1_,&buff_blob_2_);
		time_resize += timer_resize.MicroSeconds();
		timer_crop_2.Start();
		caffe::CropBlobs_gpu(buff_blob_2_ , 0, 0, 0, buff_blob_2_.height(), buff_blob_2_.width(),
				this->buff_map_, 0, rects[i].left_top.y, rects[i].left_top.x);
		time_crop_2 += timer_crop_2.MicroSeconds();
	}
	if(this->show_time_)
		LOG(INFO)<<"Time for PatchWork(in ImgPacking): "<<timer.MicroSeconds()/1000<<
			" milliseconds,in which crop_1 takes "<< time_crop_1/1000<<
			" , resize takes: "<<time_resize/1000 <<"  and crop_2 takes "<<time_crop_2/1000;
}





#ifdef CPU_ONLY
STUB_GPU(PyramidImageDataLayer);
#endif


INSTANTIATE_CLASS(BlockPacking);
INSTANTIATE_CLASS(RoiRect);
INSTANTIATE_CLASS(RectBlockPacking);
INSTANTIATE_CLASS(PyramidImageDataLayer);
INSTANTIATE_STRUCT(PyramidImageDataParam);
REGISTER_LAYER_CLASS(PyramidImageData);

//INSTANTIATE_LAYER_GPU_FORWARD(PyramidImageDataLayer);

}  // namespace caffe

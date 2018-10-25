#include <string>
#include <vector>
#include <algorithm>
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/landmark_detection_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

namespace caffe {


template<typename Dtype>
void FixedSizeBlockPacking<Dtype>::ImgPacking_cpu(const Blob<Dtype>& blob_in, RoiRect<Dtype>& roi,
		Blob<Dtype>& blob_out, int out_num_id){
	CHECK_LT(out_num_id, blob_out.num());
	CHECK_EQ(blob_out.channels(), blob_in.channels());
	CHECK_EQ(blob_out.height(), block_h_);
	CHECK_EQ(blob_out.width(), block_w_);
	buff_blob_1_.Reshape(1,blob_in.channels(),roi.height_,roi.width_ );
	CHECK_LT(std::abs(roi.scale_ -Dtype(block_h_)/roi.height_),0.0001);
	buff_blob_2_.Reshape(1,blob_in.channels(),block_h_, block_w_);
	caffe::CropBlobs_cpu(blob_in, 0,roi.start_y_,roi.start_x_,
			roi.start_y_ + roi.height_,roi.start_x_+ roi.width_,buff_blob_1_,0,0,0);
	caffe::ResizeBlob_cpu(&buff_blob_1_,&buff_blob_2_);
	memcpy(blob_out.mutable_cpu_data() + blob_out.offset(out_num_id,0,0,0),
			buff_blob_2_.cpu_data(),sizeof(Dtype)*buff_blob_2_.count());
}


template<typename Dtype>
void FixedSizeBlockPacking<Dtype>::ImgPacking_gpu(const Blob<Dtype>& blob_in, RoiRect<Dtype>& roi,
		Blob<Dtype>& blob_out, int out_num_id){

	CHECK_EQ(blob_out.channels(), blob_in.channels());
	CHECK_EQ(blob_out.height(), block_h_);
	CHECK_EQ(blob_out.width(), block_w_);
	buff_blob_1_.Reshape(1,blob_in.channels(),roi.height_,roi.width_ );
	CHECK_LT(std::abs(roi.scale_ -Dtype(block_h_)/roi.height_),0.0001);
	buff_blob_2_.Reshape(1,blob_in.channels(),block_h_, block_w_);
	caffe::CropBlobs_gpu(blob_in, 0,roi.start_y_,roi.start_x_,
			roi.start_y_ + roi.height_,roi.start_x_+ roi.width_,buff_blob_1_,0,0,0);
	caffe::ResizeBlob_gpu(&buff_blob_1_,&buff_blob_2_);
	cudaMemcpy(blob_out.mutable_gpu_data() + blob_out.offset(out_num_id,0,0,0),
			buff_blob_2_.cpu_data(),sizeof(Dtype)*buff_blob_2_.count(),cudaMemcpyDefault);
}

template<typename Dtype>
void FixedSizeBlockPacking<Dtype>::GetInputImgCoords(const RoiRect<Dtype>& roi,
		const Dtype buff_img_y, const Dtype buff_img_x, Dtype& input_y, Dtype& input_x){
	input_y = roi.start_y_ + buff_img_y/roi.scale_;
	input_x = roi.start_x_ + buff_img_x/roi.scale_;
}

template<typename Dtype>
int FixedSizeBlockPacking<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start , vector<RoiRect<Dtype> >& rois){
	int param_size = 2;
	CHECK_GE(blob.count(), start+param_size + rois.size() * 5 + 1);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = block_h_;
	*(blob_cpu_data++) = block_w_;

	*(blob_cpu_data++) = rois.size();
	for(int i=0; i <rois.size(); ++i ){
		RoiRect<Dtype> cur_roi = rois[i];
		*(blob_cpu_data++) = cur_roi.start_x_;
		*(blob_cpu_data++) = cur_roi.start_y_;
		*(blob_cpu_data++) = cur_roi.width_;
		*(blob_cpu_data++) = cur_roi.height_;
		*(blob_cpu_data++) = cur_roi.scale_;
	}
	return start+param_size + rois.size() * 5 + 1;
}


template<typename Dtype>
int FixedSizeBlockPacking<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start , vector<RoiRect<Dtype> >& rois){
	int param_size = 2;
	CHECK_GE(blob.count(), start+param_size + rois.size() * 5);
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	rois.clear();
	block_h_ = *(blob_cpu_data++);
	block_w_ = *(blob_cpu_data++);

	int n_rois = *(blob_cpu_data++);
	for(int i= 0; i < n_rois; ++i){
		RoiRect<Dtype> cur_roi;
		cur_roi.start_x_ = *(blob_cpu_data++);
		cur_roi.start_y_ = *(blob_cpu_data++);
		cur_roi.width_ = *(blob_cpu_data++);
		cur_roi.height_ = *(blob_cpu_data++);
		cur_roi.scale_ = *(blob_cpu_data++);
		rois.push_back(cur_roi);
	}
	return start+param_size + rois.size() * 5 + 1;
}

template<typename Dtype>
int LandmarkDetectionDataParam<Dtype>::SerializeToBlob(Blob<Dtype>& blob, int start ){
	int param_size = 5;
	CHECK_GE(blob.count(), start+param_size );
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	*(blob_cpu_data++) = heat_map_a_;
	*(blob_cpu_data++) = heat_map_b_;
	*(blob_cpu_data++) = mean_bgr_[0];
	*(blob_cpu_data++) = mean_bgr_[1];
	*(blob_cpu_data++) = mean_bgr_[2];
	return fixed_size_block_packer_.SerializeToBlob(blob,param_size,rois_);
}

template<typename Dtype>
int LandmarkDetectionDataParam<Dtype>::ReadFromSerialized(Blob<Dtype>& blob, int start ){
	int param_size = 5;
	CHECK_GE(blob.count(), start+param_size );
	Dtype* blob_cpu_data = blob.mutable_cpu_data() + start;
	heat_map_a_ = *(blob_cpu_data++) ;
	heat_map_b_ = *(blob_cpu_data++) ;
	mean_bgr_[0] = *(blob_cpu_data++) ;
	mean_bgr_[1] = *(blob_cpu_data++) ;
	mean_bgr_[2] = *(blob_cpu_data++);
	return fixed_size_block_packer_.ReadFromSerialized(blob,param_size,rois_);
}

template<typename Dtype>
void LandmarkDetectionDataParam<Dtype>::SetUpParam(const LandmarkDetectionDataParameter & landmark_detection_data_param){
	this->heat_map_a_ = landmark_detection_data_param.heat_map_a();
	this->heat_map_b_ = landmark_detection_data_param.heat_map_b();
	CHECK_GT(heat_map_a_,0);
	mean_bgr_[0] = landmark_detection_data_param.mean_b();
	mean_bgr_[1] = landmark_detection_data_param.mean_g();
	mean_bgr_[2] = landmark_detection_data_param.mean_r();
	LOG(INFO)<<"Mean_bgr=["<<mean_bgr_[0]<<" "<<mean_bgr_[1] <<" "<<mean_bgr_[2]<<"].";
	this->block_h_ = landmark_detection_data_param.block_h();
	this->block_w_ = landmark_detection_data_param.block_w();
	fixed_size_block_packer_.setBlockH(this->block_h_);
	fixed_size_block_packer_.setBlockW(this->block_w_);

}

template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	const LandmarkDetectionDataParameter & landmark_detection_data_param =
			this->layer_param_.landmark_detection_data_param();
	this->SetUpParam(landmark_detection_data_param);
	CHECK(landmark_detection_data_param.has_num_anno_points_per_instance());
	num_anno_points_per_instance_ = landmark_detection_data_param.num_anno_points_per_instance();

	CHECK(landmark_detection_data_param.has_roi_center_point());
	CHECK(landmark_detection_data_param.has_standard_len_point_1());
	CHECK(landmark_detection_data_param.has_standard_len_point_2());
	CHECK(landmark_detection_data_param.has_standard_len());
	this->roi_center_point_ = landmark_detection_data_param.roi_center_point();
	this->standard_len_point_1_ = landmark_detection_data_param.standard_len_point_1();
	this->standard_len_point_2_ = landmark_detection_data_param.standard_len_point_2();
	this->standard_len_ = landmark_detection_data_param.standard_len();
	this->min_valid_standard_len_ = landmark_detection_data_param.min_valid_standard_len();
	this->restrict_roi_in_center_ = landmark_detection_data_param.restrict_roi_in_center();

	count_ = 0;

	const FCNImageDataSourceParameter& data_source_param = landmark_detection_data_param.data_source_param();
	data_provider_.SetUpParameter(data_source_param);
	data_provider_.ReadPosAndNegSamplesFromFiles(data_source_param,
				this->num_anno_points_per_instance_,-1);
	this->batch_size_ = data_provider_.GetBatchSize();
// @TODO get batch_num_for_one_epoch

	LOG(INFO)<<"fixed_size_block_packer_: h:"<<this->fixed_size_block_packer_.block_h()
			<<", w:"<<this->fixed_size_block_packer_.block_w();
	buffered_reader_.reset(new BufferedColorJPGReader<Dtype>("",5));



//	LOG(INFO)<<"shape of data_:"<<batch_size_<<", "<<3<<", "<<block_h_<<", "<<block_w_;
	this->prefetch_data_.Reshape(batch_size_,3,this->block_h_,this->block_w_);
	top[0]->Reshape(batch_size_,3,this->block_h_,this->block_w_);
	this->prefetch_label_.Reshape(1,1,60,1000);
	top[1]->Reshape(1,1,60,1000);

	show_output_path_ = "cache/LandmarkDetectionDataLayer";
	char output_path[512];
	this->pic_print_ = ((getenv("PIC_PRINT") != NULL) && (getenv("PIC_PRINT")[0] == '1'));
	if (this->pic_print_) {
			sprintf(output_path, "%s/pic",this->show_output_path_.c_str());
			CreateDir(output_path);
	}

}


template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>::InternalThreadEntry(){
//	LOG(INFO)<<"intput start" ;
	this->prefetch_rois_.clear();
	data_provider_.FetchBatchSamples();
	prefetch_batch_samples_ = data_provider_.cur_batch_samples();
	for(int i=0; i < batch_size_; ++i){
		string img_name = prefetch_batch_samples_[i].first;
		vector<Dtype>& coords =  prefetch_batch_samples_[i].second;
//		LOG(INFO)<<"loading image: "<<img_name;
		img_name = img_name+ ".jpg";
		buffered_reader_->LoadToBlob(img_name.c_str(),image_blob_);
//		LOG(INFO)<<"loading image finished" ;
		int lt_x,lt_y,rb_x,rb_y;
		GetRoi(image_blob_.width(), image_blob_.height(),coords,lt_x,lt_y,rb_x,rb_y);
//		LOG(INFO)<<"GetROI finished with coords: "<<lt_x<<" "<<lt_y<<" "<<rb_x<<" "<<rb_y ;
		RoiRect<Dtype> roi(Dtype(this->block_h_)/(rb_y - lt_y),Dtype(lt_y), Dtype(lt_x),Dtype(rb_y - lt_y),Dtype(rb_x - lt_x) );
		this->fixed_size_block_packer_.ImgPacking_cpu(image_blob_,roi,this->prefetch_data_,i);
//		LOG(INFO)<<"ImagePack finished" ;
		ImgBlobRemoveMean_cpu( this->prefetch_data_,
						this->mean_bgr_[0], this->mean_bgr_[1], this->mean_bgr_[2],0,i);
		this->prefetch_rois_.push_back(roi);
	}

	if (this->pic_print_){
		ShowImg(this->prefetch_data_);
	}
//	LOG(INFO)<<"intput finished" ;
}

template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>:: Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
//	LOG(INFO)<<"forward start";
	this->JoinPrefetchThread();

//	LOG(INFO)<<"after joint thread start";
	this->batch_samples_ = this->prefetch_batch_samples_;
	this->rois_ = this->prefetch_rois_;
	this->SerializeToBlob(*(top[1]),0);
	caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
	             top[0]->mutable_cpu_data());
//	caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
//				 top[1]->mutable_cpu_data());
	this->CreatePrefetchThread();
}

template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>:: Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}


template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>::ShowImg(const Blob<Dtype>& top){
	char path[512];
	const Dtype* top_data = top.cpu_data();
	for(int i=0; i < batch_size_; ++i){
		pair<string, vector<Dtype> >&  cur_sample_ = this->prefetch_batch_samples_[i];
		string img_name = ImageDataSourceProvider<Dtype>::GetSampleName(cur_sample_);
		cv::Mat packed_img = BlobImgDataToCVMat(top,i,this->mean_bgr_[0],this->mean_bgr_[1],this->mean_bgr_[2]);
		sprintf(path, "%s/pic/%08d_%s_cropped_img.jpg", show_output_path_.c_str(),count_++,img_name.c_str());
//		LOG(INFO)<<"saving "<<path;
		imwrite(path,packed_img);
	}
}


template<typename Dtype>
void LandmarkDetectionDataLayer<Dtype>::GetRoi(const int img_w, const int img_h,
		const vector<Dtype>& coords,int& lt_x, int& lt_y, int& rb_x, int& rb_y){
	Dtype sample_scales  = 1;
	lt_x = lt_y = -1;
	rb_x = rb_y = 1;
	if (coords[standard_len_point_1_ * 2] != -1 && coords[standard_len_point_2_ * 2] != -1 &&
		coords[standard_len_point_1_ * 2 +1] != -1 && coords[standard_len_point_2_ * 2+1] != -1)
	{
		const int len = standard_len_;
		float x_diff = fabs(coords[standard_len_point_2_ * 2] - coords[standard_len_point_1_ * 2]);
		float y_diff = fabs(coords[standard_len_point_2_ * 2 + 1] - coords[standard_len_point_1_ * 2 + 1]);
		float pt_diff = sqrtf(x_diff * x_diff + y_diff * y_diff);
		if(pt_diff >= min_valid_standard_len_ ){
			Dtype tmp_scale =   len / pt_diff  ;
			sample_scales *= tmp_scale;
		}else {
			LOG(INFO)<<"Warning: Standard distance is too small. pt_diff = "<< pt_diff <<" , pt1 = ("<<
					coords[standard_len_point_1_ * 2]<<","<<coords[standard_len_point_1_ * 2 +1]<<") , pt2 = ("<<
					coords[standard_len_point_2_ * 2]<<","<<coords[standard_len_point_2_ * 2+1]<<").";
//			return;
		}

		Dtype center_x =  coords[roi_center_point_ * 2] ;
		Dtype center_y =  coords[roi_center_point_ * 2 +1 ] ;

		int patch_height = round(this->block_h_ / sample_scales );
		int patch_width = round(this->block_w_ / sample_scales );
		int lt_x_added = MAX(0, round(center_x  + patch_width/2   ) - img_w);
		lt_x =  MIN(img_w , MAX(0,round(center_x - patch_width/2 )-lt_x_added));

		int lt_y_added = MAX(0, round(center_y + patch_height/2  ) - img_h);
		lt_y = MIN(img_h ,MAX(0,round(center_y - patch_height/2 )-lt_y_added ));

		int rb_x_added = MAX(0, 0 - round( center_x - patch_width/2 ));
		rb_x =  MIN(img_w ,rb_x_added + MAX(0,round(center_x + patch_width/2 )));

		int rb_y_added = MAX(0, 0 - round(center_y - patch_height/2  ));
		rb_y = MIN(img_h ,rb_y_added + MAX(0,round(center_y + patch_height/2  )));

		if(restrict_roi_in_center_){
			lt_x =  round(center_x - patch_width/2  );
			lt_y =	round(center_y - patch_height/2 );
			rb_x =  round(center_x + patch_width/2  );
			rb_y = 	round(center_y + patch_height/2  );
		}
		CHECK_GT(rb_y - lt_y,0);
		CHECK_GT(rb_x - lt_x,0);
	}

}


/**
 * For LandmarkDetectionOutputLayer
 */

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
	const LandmarkDetectionOutputParameter & landmark_detection_output_param =
				this->layer_param_.landmark_detection_output_param();
	threshold_ = landmark_detection_output_param.threshold();
	channel_per_scale_ = landmark_detection_output_param.channel_per_scale();
	channel_per_point_ = landmark_detection_output_param.channel_per_point();
	CHECK_EQ(channel_per_scale_ % channel_per_point_, 0);
	CHECK_EQ(channel_per_point_, 3);
	nms_need_nms_ = landmark_detection_output_param.nms_param().need_nms();
	nms_dist_threshold_ = landmark_detection_output_param.nms_param().dist_threshold();
	nms_top_n_ = landmark_detection_output_param.nms_param().top_n();
	nms_add_score_ = landmark_detection_output_param.nms_param().add_score();

	line_point_pairs_.clear();
	if(landmark_detection_output_param.has_draw_line_point_pair_file()){
		std::ifstream key_points_file(landmark_detection_output_param.draw_line_point_pair_file().c_str());
		CHECK(key_points_file.good()) << "Failed to open line point pair file "
				<< landmark_detection_output_param.draw_line_point_pair_file() << std::endl;
		std::ostringstream oss;
		int key_point;
		while(key_points_file >> key_point) {
			CHECK_LT(key_point, bottom[0]->channels());
			this->line_point_pairs_.push_back(key_point);
			oss <<"["<< key_point << " ";
			key_points_file >> key_point;
			CHECK_LT(key_point, bottom[0]->channels());
			this->line_point_pairs_.push_back(key_point);
			oss << key_point << " ";
			oss <<"] ";
		}
		key_points_file.close();
		CHECK_EQ(line_point_pairs_.size()%2, 0);
		LOG(INFO) << "There are " << line_point_pairs_.size()/2 << "pairs of line points: " << oss.str();
	}

	show_output_path_ = "cache/LandmarkDetectionOutputLayer";
	char output_path[512];
	this->pic_print_ = ((getenv("PIC_PRINT") != NULL) && (getenv("PIC_PRINT")[0] == '1'));
	if (this->pic_print_) {
			sprintf(output_path, "%s/pic",this->show_output_path_.c_str());
			CreateDir(output_path);
	}

	count_ = 0;
}



template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
//	LOG(INFO)<<"forward end";
	landmark_detection_data_param_.ReadFromSerialized(*(bottom[1]), 0);
//	const int n_point = bottom[0]->channels();
	CHECK_EQ(bottom[0]->channels()%this->channel_per_scale_, 0);
	const int n_num = bottom[0]->num();
	Dtype center_x = bottom[0]->width() * landmark_detection_data_param_.heat_map_a_;
	Dtype center_y = bottom[0]->height() * landmark_detection_data_param_.heat_map_a_;
	while(output_points_.size() < n_num){
		output_points_.push_back(vector< ScorePoint<Dtype> >());
	}


//	const Dtype* out_data = bottom[0]->cpu_data();
//	int out_dim = bottom[0]->count()/n_num;
//	const Dtype* in_data = bottom[2]->cpu_data();
//	int in_dim = bottom[2]->count()/n_num;
//	int in_height = bottom[2]->height();
//	int in_width = bottom[2]->width();
	for(int i=0; i < n_num; ++i){

//		std::cout<<"out_data of img_id = "<<i<<std::endl;
//		for(int d=0; d < out_dim; ++d){
//			std::cout<<out_data[bottom[0]->offset(i) + d]<<" ";
//		}
//		std::cout<<std::endl;
//
//		std::cout<<"in_data of img_id = "<<i<<std::endl;
//		for(int h=0; h < in_height; ++h){
//			for(int d=0; d < in_width; ++d){
//				std::cout<<in_data[bottom[2]->offset(i,0,h,d) ]<<" ";
//				std::cout<<in_data[bottom[2]->offset(i,1,h,d) ]<<" ";
//				std::cout<<in_data[bottom[2]->offset(i,2,h,d) ]<<" ";
//			}
//		}
//		std::cout<<std::endl;

//		CHECK(false);
		vector< vector< ScorePoint<Dtype> > > un_filtered_points_of_one_face;
		GetUnFilteredPointsForOneFaceInBlob(*bottom[0],i,un_filtered_points_of_one_face);
		FilterPointsForOneFace(un_filtered_points_of_one_face,output_points_[i],center_x, center_y);
		/**
		 * @Todo  ShowImg
		 */
	}
	if(this->pic_print_){
		CHECK_EQ(bottom.size(),3);
		ShowImg(*bottom[2],output_points_);
	}

	for(int i=0; i < n_num; ++i){
		BlockCoordsToImgCoords(this->landmark_detection_data_param_.rois_[i],output_points_[i]);
	}
	count_ += n_num;
//	std::cout<<" output count "<< count_<<" finished"<<std::endl;
}

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::ShowImg(const Blob<Dtype>& blob,
		vector<vector<ScorePoint<Dtype> > >&  all_filtered_face_points){
	const int n_num = blob.num();
	char path[512];
	for(int n_id = 0; n_id < n_num; ++n_id){
		cv::Mat mat = BlobImgDataToCVMat(blob,n_id, landmark_detection_data_param_.mean_bgr_[0],
			landmark_detection_data_param_.mean_bgr_[1],landmark_detection_data_param_.mean_bgr_[2]);
		DrawScorePointsOnMat(mat, all_filtered_face_points[n_id], cv::Scalar(255,255,0),1,line_point_pairs_);
		sprintf(path, "%s/pic/%06d_filtered_face_points.jpg", show_output_path_.c_str(),count_+n_id);
//		LOG(INFO)<<"saving "<<path;
		imwrite(path,mat);
	}
}

//template<typename Dtype>
//void LandmarkDetectionOutputLayer<Dtype>::BlobAndResultToCvMat(const Blob<Dtype>& blob,
//		const int num_id, vector<ScorePoint<Dtype> >  & face_points_, cv::Mat& mat){
//
//}

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::ShowImg(const Blob<Dtype>& blob, const int num_id,
		vector<vector<ScorePoint<Dtype> > >&  un_filtered_face_points_){
	cv::Mat mat = BlobImgDataToCVMat(blob,num_id, landmark_detection_data_param_.mean_bgr_[0],
			landmark_detection_data_param_.mean_bgr_[1],landmark_detection_data_param_.mean_bgr_[2]);
	vector<int> empty_pair;
	for(int i = 0; i < un_filtered_face_points_.size(); ++i){
		DrawScorePointsOnMat(mat, un_filtered_face_points_[i], cv::Scalar(255,255,0),1,empty_pair);
	}
	char path[512];
	sprintf(path, "%s/pic/%06d_un_filtered_face_points.jpg", show_output_path_.c_str(),count_+num_id);
	LOG(INFO)<<"saving "<<path;
	imwrite(path,mat);
}

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::GetUnFilteredPointsForOneFaceInBlob(Blob<Dtype>& blob,
		int num_id,vector<vector<ScorePoint<Dtype> > >& res){
	res.clear();
	CHECK_LT(num_id,blob.num());
	RoiRect<Dtype> cur_roi = landmark_detection_data_param_.rois_[num_id];
	Dtype* bottom_data = blob.mutable_cpu_data();
	const int map_height = blob.height();
	const int map_width  = blob.width();
	const int map_size = map_height* map_width;
	const int n_points = channel_per_scale_ / channel_per_point_;
	const int scale_num = blob.channels()/ channel_per_scale_;
	const int heatmap_a = landmark_detection_data_param_.heat_map_a_;
	const int heatmap_b = landmark_detection_data_param_.heat_map_b_;
//	LOG(INFO)<<"heatmap_a:"<<heatmap_a<<",  heatmap_b:"<<heatmap_b;
	for(int point_id=0; point_id < n_points; ++point_id){
		vector<ScorePoint<Dtype> > cur_point_candidates;
		cur_point_candidates.clear();
		for(int scale_id = 0; scale_id < scale_num; ++scale_id){
			int score_channel = scale_id*channel_per_scale_ + point_id;
			int offset_channel = scale_id*channel_per_scale_ + n_points + point_id*2;
			Dtype* scores = bottom_data + blob.offset(num_id,score_channel  , 0,0);
			Dtype* dx1 =    bottom_data + blob.offset(num_id,offset_channel+0,0,0);
			Dtype* dy1 =    bottom_data + blob.offset(num_id,offset_channel+1,0,0);
			for(int off = 0; off< map_size; ++off){
				if(scores[off] < this->threshold_)
					continue;
				int h = off / map_width;
				int w = off % map_width ;
				Dtype coord_x = ((w - dx1[off])*heatmap_a + heatmap_b);
				Dtype coord_y = ((h - dy1[off])*heatmap_a + heatmap_b);
				ScorePoint<Dtype> score_point(scores[off],coord_x,coord_y);
				cur_point_candidates.push_back(score_point);
			}
		}
//		LOG(INFO)<<"before nms: "<< cur_point_candidates[0].score <<
//				" "<<cur_point_candidates[0].x <<" "<<cur_point_candidates[0].y;
		vector<bool> is_selected = caffe::nms(cur_point_candidates,this->nms_dist_threshold_,this->nms_top_n_);
		vector<ScorePoint<Dtype> > cur_selected_point_candidates;
		for(int i=0; i < is_selected.size(); ++i){
			if(is_selected[i]){
				cur_selected_point_candidates.push_back(cur_point_candidates[i]);
			}
		}
//		LOG(INFO)<<"after nms: "<< cur_selected_point_candidates[0].score <<
//						" "<<cur_selected_point_candidates[0].x <<" "<<cur_selected_point_candidates[0].y;
		res.push_back(cur_selected_point_candidates);
	}
}

/**
 *  Current stragegy: get the nearest points from center.
 */
template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::FilterPointsForOneFace(vector<vector<ScorePoint<Dtype> > >& res,
		vector<ScorePoint<Dtype> > & dst, Dtype center_x, Dtype center_y){
	dst.resize(res.size());
	ScorePoint<Dtype> center(1,center_x, center_y);
	for(int point_id = 0; point_id < res.size(); ++point_id){
		ScorePoint<Dtype>& point_res = dst[point_id];
		point_res.score = -1;
		point_res.x = 0;
		point_res.y = 0;
		vector<ScorePoint<Dtype> >& cur_points = res[point_id];
		Dtype min_dist = 1000000000;
		for(int i=0; i < cur_points.size(); ++i){
			Dtype dist = center.dist(cur_points[i]);
			if(dist < min_dist){
				min_dist = dist;
				point_res = cur_points[i];
			}
		}
	}
}

template<typename Dtype>
void LandmarkDetectionOutputLayer<Dtype>::BlockCoordsToImgCoords(const RoiRect<Dtype>& roi,
		vector<ScorePoint<Dtype> >& filtered_points){
	for(int i=0; i< filtered_points.size(); ++i){
		ScorePoint<Dtype>& point = filtered_points[i];
		point.x =  point.x/roi.scale_ + roi.start_x_;
		point.y =  point.y/roi.scale_ + roi.start_y_;
	}
}


INSTANTIATE_CLASS(FixedSizeBlockPacking);
INSTANTIATE_CLASS(LandmarkDetectionDataLayer);
INSTANTIATE_CLASS(LandmarkDetectionOutputLayer);

REGISTER_LAYER_CLASS(LandmarkDetectionData);
REGISTER_LAYER_CLASS(LandmarkDetectionOutput);

#ifdef CPU_ONLY
STUB_GPU(LandmarkDetectionDataLayer);
STUB_GPU(LandmarkDetectionOutputLayer);
#endif



}  // namespace caffe

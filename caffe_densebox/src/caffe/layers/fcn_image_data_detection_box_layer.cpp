#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/rng.hpp"
#include <sys/param.h>
#include <iomanip>
namespace caffe {

template <typename Dtype>
IImageDataDetectionBox<Dtype>::IImageDataDetectionBox()
{
	channel_detection_base_channel_offset_ = 0;
	channel_detection_label_channel_offset_ =0;
	ignore_class_flag_id_ = -1;
}


template <typename Dtype>
void IImageDataDetectionBox<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_image_data_param);
	CHECK(fcn_image_data_param.has_fcn_image_data_detection_box_param())<<
		"fcn_image_data_detection_box_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_detection_box_param());
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>::SetUpParameter(
		const FCNImageDataDetectionBoxParameter& fcn_img_data_detection_box_param)
{
	this->min_output_pos_radius_ = fcn_img_data_detection_box_param.min_output_pos_radius();
	this->ignore_margin_ = fcn_img_data_detection_box_param.ignore_margin();
	/**
	 * Set param for bbox prediction
	 */
	this->ignore_class_flag_id_ = fcn_img_data_detection_box_param.ignore_class_flag_id();
	this->loc_regress_on_ignore_ = fcn_img_data_detection_box_param.loc_regress_on_ignore();
	this->bbox_height_ = fcn_img_data_detection_box_param.bbox_height();
	this->bbox_width_ = fcn_img_data_detection_box_param.bbox_width();
	this->bbox_size_norm_type_ = fcn_img_data_detection_box_param.bbox_size_norm_type();
	this->bbox_valid_dist_ratio_ = fcn_img_data_detection_box_param.bbox_valid_dist_ratio();
	CHECK(this->bbox_valid_dist_ratio_ > 0 )<<
			"bbox_valid_dist_ratio should be in the range of (0,+oo)";
	this->bbox_point_id_.clear();
	std::copy(fcn_img_data_detection_box_param.bbox_point_id().begin(),
			fcn_img_data_detection_box_param.bbox_point_id().end(),
			std::back_inserter(this->bbox_point_id_));
	CHECK(this->bbox_point_id_.size() == 2)<<"this->bbox_point_id_.size() == "<<this->bbox_point_id_.size()
				  <<" bbox point number should be 2.";
	LOG(INFO)<<"Detection label is activated. bbox_size=[" << this->bbox_height_<<
			", "<<this->bbox_width_<< "], the bbox_point size ="<<2;

	this->need_detection_loc_diff_ = fcn_img_data_detection_box_param.need_detection_loc_diff();
	LOG(INFO)<<"need_detection_loc_diff_ == "<<need_detection_loc_diff_;
	this->bbox_loc_diff_valid_dist_ratio_ = fcn_img_data_detection_box_param.bbox_loc_diff_valid_dist_ratio();
	this->total_class_num_ = fcn_img_data_detection_box_param.total_class_num();
	if(total_class_num_ == 1 && fcn_img_data_detection_box_param.has_class_flag_id()  ){
		class_flag_id_ = -1;
		LOG(INFO)<<"if total_class_num == 1, there's no need to setup class_flag_id";
	}
	if(total_class_num_ > 1){
		CHECK(fcn_img_data_detection_box_param.has_class_flag_id());
		class_flag_id_ = fcn_img_data_detection_box_param.class_flag_id();
	}
	/**
	 * Set param for point_diff_from_center
	 */
	this->point_id_for_point_diff_from_center_.clear();
	this->point_ignore_flag_id_for_point_diff_from_center_.clear();
	this->need_point_diff_from_center_ = fcn_img_data_detection_box_param.has_list_point_diff_from_center();
  this->has_point_ignore_flag_diff_from_center_ =
  		fcn_img_data_detection_box_param.has_list_point_ignore_flag_diff_from_center();
	if(need_point_diff_from_center_){
		std::ifstream key_points_file(fcn_img_data_detection_box_param.list_point_diff_from_center().c_str());
		CHECK(key_points_file.good()) << "Failed to open point diff from center file "
				<< fcn_img_data_detection_box_param.list_point_diff_from_center() << std::endl;
		std::ostringstream oss;
		int key_point;
		while(key_points_file >> key_point) {
		  this->point_id_for_point_diff_from_center_.push_back(key_point);
		  CHECK_LT(key_point,this->num_anno_points_per_instance_) << "key point id for point_diff_from_center "<<
				  "could not exceed the size of number of total points for one instance";
		  oss << key_point << " ";
		}
		key_points_file.close();
		LOG(INFO) << "There are " << point_id_for_point_diff_from_center_.size()
		<< " points need location diff from center: " << oss.str();

		if(this->has_point_ignore_flag_diff_from_center_){
			std::ifstream key_points_file(fcn_img_data_detection_box_param.list_point_ignore_flag_diff_from_center().c_str());
			CHECK(key_points_file.good()) << "Failed to open point ignore flag diff from center file "
					<< fcn_img_data_detection_box_param.list_point_ignore_flag_diff_from_center() << std::endl;
			std::ostringstream oss;
			int key_point;
			while(key_points_file >> key_point) {
				this->point_ignore_flag_id_for_point_diff_from_center_.push_back(key_point);
				CHECK_LT(key_point,this->num_anno_points_per_instance_) << "key point id for point_ignore_flag_diff_from_center "<<
						"could not exceed the size of number of total points for one instance";
				oss << key_point << " ";
			}
			key_points_file.close();
			LOG(INFO) << "There are " << point_ignore_flag_id_for_point_diff_from_center_.size()
					<< " points ignore flag of diff from center: " << oss.str();
			CHECK_EQ(this->point_ignore_flag_id_for_point_diff_from_center_.size(),this->point_id_for_point_diff_from_center_.size());
		}
	}
	need_point_diff_from_center_ = point_id_for_point_diff_from_center_.size() > 0;
	LOG(INFO)<<"need_point_diff_from_center_ == "<<need_point_diff_from_center_;

	gt_bboxes_.clear();
	bbox_print_ = ((getenv("BBOX_PRINT") != NULL) && (getenv("BBOX_PRINT")[0] == '1'));
	char output_path[512];
	if (this->pic_print_) {
		sprintf(output_path, "%s/pic",this->show_output_path_.c_str());
		CreateDir(output_path);
	}
	if (this->label_print_) {
		sprintf(output_path, "%s/label",this->show_output_path_.c_str());
		CreateDir(output_path);
	}
}

template <typename Dtype>
int IImageDataDetectionBox<Dtype>::SetUpChannelInfo( const int channel_base_offset )
{
	channel_detection_base_channel_offset_ = channel_base_offset;
	const int n_scale_base = this->scale_base_.size();
	const int n_diff_from_center_points = point_id_for_point_diff_from_center_.size();
	channel_detection_ignore_label_channel_offset_ = 1 * n_scale_base * total_class_num_;
	channel_detection_diff_channel_offset_ = channel_detection_ignore_label_channel_offset_ + 1 * n_scale_base * total_class_num_;
	channel_point_diff_from_center_channel_offset_ = channel_detection_diff_channel_offset_+ need_detection_loc_diff_ * 4 * n_scale_base * total_class_num_;
	channel_detection_channel_all_need_ = channel_point_diff_from_center_channel_offset_ + n_diff_from_center_points*
			(3+this->has_point_ignore_flag_diff_from_center_)*n_scale_base*total_class_num_;

	channel_point_score_channel_offset_ = channel_point_diff_from_center_channel_offset_;
	channel_point_ignore_channel_offset_ = channel_point_score_channel_offset_ + n_diff_from_center_points* n_scale_base*total_class_num_;
	channel_point_diff_channel_offset_ = channel_point_ignore_channel_offset_ + n_diff_from_center_points*
			n_scale_base*total_class_num_ *  has_point_ignore_flag_diff_from_center_;

	LOG(INFO)<<"total number of label channels for detection: "<<channel_detection_channel_all_need_ <<", in which confidence labels need "<<
			channel_detection_ignore_label_channel_offset_ - channel_detection_label_channel_offset_ <<" , and ignore labels need "<<
			channel_detection_diff_channel_offset_ - channel_detection_ignore_label_channel_offset_<<" , and detection diffs need " <<
			channel_point_diff_from_center_channel_offset_ - channel_detection_diff_channel_offset_ <<" , and point_diff_from_center need "<<
			channel_detection_channel_all_need_ - channel_point_diff_from_center_channel_offset_;
	this->total_channel_need_ = channel_detection_base_channel_offset_ + channel_detection_channel_all_need_;
	return channel_detection_base_channel_offset_ + channel_detection_channel_all_need_;
}



template <typename Dtype>
vector<Dtype> IImageDataDetectionBox<Dtype>::GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance)
{
	vector<Dtype> scales ;
	int num_instances =  coords_of_all_instance.size() / (this->num_anno_points_per_instance_ * 2);
	CHECK(coords_of_all_instance.size() % (this->num_anno_points_per_instance_ * 2) == 0);
	CHECK_GT(num_instances,0);
	int box_idx1 = this->bbox_point_id_[0] * 2;
	int box_idx2 = this->bbox_point_id_[1] * 2;

	for(int i = 0; i < num_instances; ++i )
	{

		int cur_bbox_idx1 = box_idx1 + this->num_anno_points_per_instance_ * 2 * i ;
		int cur_bbox_idx2 = box_idx2 + this->num_anno_points_per_instance_ * 2 * i ;

		Dtype cur_height = coords_of_all_instance[cur_bbox_idx2 + 1] - coords_of_all_instance[cur_bbox_idx1 + 1];
		Dtype cur_width = coords_of_all_instance[cur_bbox_idx2] - coords_of_all_instance[cur_bbox_idx1];
		Dtype cur_diag = sqrt(cur_height*cur_height + cur_width*cur_width);

		switch(bbox_size_norm_type_)
		{
			case FCNImageDataDetectionBoxParameter_BBoxSizeNormType_HEIGHT:
			{
				scales.push_back(cur_height/this->bbox_height_);
				break;
			}
			case FCNImageDataDetectionBoxParameter_BBoxSizeNormType_WIDTH:
			{
				scales.push_back(cur_width/this->bbox_width_);
				break;
			}
			case FCNImageDataDetectionBoxParameter_BBoxSizeNormType_DIAG:
			{
				scales.push_back(cur_diag/(sqrt(bbox_height_ * bbox_height_ +
						bbox_width_ * bbox_width_)));
				break;
			}
			default:
				LOG(FATAL) << "Unknown type " << bbox_size_norm_type_;
		}
	}
	return scales;
}


template <typename Dtype>
void IImageDataDetectionBox<Dtype>::GTBBoxesToBlob(Blob<Dtype>& prefetch_bbox){
	int n_bbox = gt_bboxes_.size();
	prefetch_bbox.Reshape(0, 0,0,0);
	if(n_bbox > 0){
		int bbox_data_size = gt_bboxes_[0].size();
		prefetch_bbox.Reshape(n_bbox, 1,1,bbox_data_size);
		Dtype* bbox_data = prefetch_bbox.mutable_cpu_data();
		for(int i=0; i < n_bbox; ++i){
			for(int j = 0; j < bbox_data_size; ++j){
				bbox_data[j] = gt_bboxes_[i][j];
			}
			bbox_data += bbox_data_size;
		}
	}
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>::PrintBBoxes(const Blob<Dtype>& prefetch_bbox){
	LOG(INFO)<<"The ground truth bboxes in current batch: ";
	int num = prefetch_bbox.num();
	if(num == 0 )
		return;
	const Dtype* const_bbox_data = prefetch_bbox.cpu_data();
	int data_size = prefetch_bbox.width();
	for(int i=0; i < num; ++i){
		const Dtype* bbox_data =  const_bbox_data +data_size * i;
		LOG(INFO)<<"In Pic "<<int(bbox_data[1])<<" , class "<<int(bbox_data[0]) <<" is found at ("<<
				bbox_data[2]<<","<<bbox_data[3]<<","<<bbox_data[4]<<","<<bbox_data[5]<<").";

	}
}



template <typename Dtype>
void IImageDataDetectionBox<Dtype>::GenerateDetectionMapForOneInstance(int item_id,
		const vector<Dtype> & coords_of_one_instance, const Dtype scale,
		const vector<ImageDataAnnoType> anno_type,Blob<Dtype>& prefetch_label,int used_scale_base_id)
{

	CHECK_GE(coords_of_one_instance.size(), this->num_anno_points_per_instance_ *2);

	int n_diff_from_center_points = this->point_id_for_point_diff_from_center_.size();
	Dtype* top_label =  prefetch_label.mutable_cpu_data();

	int scale_base_size = this->scale_base_.size();
	int idx1 = this->bbox_point_id_[0]*2;
	int idx2 = this->bbox_point_id_[1]*2;
	int class_id = 0;
	if(this->total_class_num_ > 1){
		int class_flag_id = this->class_flag_id_ * 2;
		CHECK_GT(coords_of_one_instance.size(), idx1+1);
		CHECK_GT(coords_of_one_instance.size(), idx2+1);
		CHECK_GT(coords_of_one_instance.size(), class_flag_id+1);
		class_id = coords_of_one_instance[class_flag_id];
	}

	CHECK_LT(class_id, this->total_class_num_);
	CHECK_GE(class_id, 0);
	if(coords_of_one_instance[idx1] != -1 && coords_of_one_instance[idx1+1] != -1
		&& coords_of_one_instance[idx2] != -1 && coords_of_one_instance[idx2+1] != -1)
	{

		/**
		 *   Push bbox to gt_bboxes_;
		 */
		vector<Dtype> bbox(6,-1);
		bbox[0] = class_id;
		bbox[1] = item_id;
		bbox[2] = (coords_of_one_instance[idx1]   - this->heat_map_b_) / this->heat_map_a_;
		bbox[3] = (coords_of_one_instance[idx1+1] - this->heat_map_b_) / this->heat_map_a_;
		bbox[4] = (coords_of_one_instance[idx2]   - this->heat_map_b_) / this->heat_map_a_;
		bbox[5] = (coords_of_one_instance[idx2+1] - this->heat_map_b_) / this->heat_map_a_;
		boost::unique_lock<boost::shared_mutex> write_lock(bbox_mutex_);
		gt_bboxes_.push_back(bbox);
		write_lock.unlock();

		Dtype mid_bbox_x = coords_of_one_instance[idx1]/2 + coords_of_one_instance[idx2]/2;
		Dtype mid_bbox_y = coords_of_one_instance[idx1 +1]/2 + coords_of_one_instance[idx2+1]/2;
		Dtype heat_map_x = (mid_bbox_x - this->heat_map_b_) / this->heat_map_a_;
		Dtype heat_map_y = (mid_bbox_y - this->heat_map_b_) / this->heat_map_a_;

		Dtype pt_x, pt_y;
		for(int scale_base_id = 0 ; scale_base_id < scale_base_size; ++ scale_base_id)
		{
			int local_channel_point_score_offset =  n_diff_from_center_points *  (scale_base_id*total_class_num_ +class_id);
			int local_channel_point_diff_offset =  n_diff_from_center_points *  (scale_base_id*total_class_num_ +class_id)*2;

			Dtype max_loc_diff_valid_distance_horizontal =MAX(  (this->bbox_width_ * this->bbox_loc_diff_valid_dist_ratio_ *scale
					 / Dtype(this->heat_map_a_)) ,min_output_pos_radius_);
			Dtype max_loc_diff_valid_distance_vertical = MAX(  (this->bbox_height_ * this->bbox_loc_diff_valid_dist_ratio_ *scale
				     / Dtype(this->heat_map_a_)) ,min_output_pos_radius_);
			Dtype max_valid_distance_horizontal =  MAX(  (this->bbox_width_ * (this->bbox_valid_dist_ratio_)*scale
					 / Dtype(this->heat_map_a_)), min_output_pos_radius_)   ;
			Dtype max_total_dist_horizontal = MAX(max_valid_distance_horizontal + ignore_margin_ *scale ,
					max_loc_diff_valid_distance_horizontal+  (ignore_margin_ *scale) )  ;
			Dtype max_valid_distance_vertical =  MAX(  (this->bbox_height_ * (this->bbox_valid_dist_ratio_)*scale
				 / Dtype(this->heat_map_a_)), min_output_pos_radius_) ;
			Dtype max_total_dist_vertical = MAX(max_valid_distance_vertical+ignore_margin_*scale ,
					max_loc_diff_valid_distance_vertical+ (ignore_margin_ *scale) )  ;

			for(int dy = -round(max_total_dist_vertical); dy <= round(max_total_dist_vertical); ++dy )
			{
				pt_y = heat_map_y + dy;
				if (round(pt_y) < 0 || round(pt_y) >=  prefetch_label.height())
						continue;
				for(int dx = -round(max_total_dist_horizontal); dx <= round(max_total_dist_horizontal); ++dx )
				{
					pt_x = heat_map_x + dx;
					if ( round(pt_x) < 0 || round(pt_x) >=  prefetch_label.width())
						continue;
					int base_offset = prefetch_label.offset(item_id, channel_detection_base_channel_offset_,round(pt_y), round(pt_x));
					/**
					 * check which type of ground truth is needed
					 */
					bool need_bbox_label_generating = (anno_type[scale_base_id] == caffe::ANNO_POSITIVE); // need positive label ( the center of bbox ) for confidense score
					bool need_bbox_loc_generating = this->need_detection_loc_diff_ && (anno_type[scale_base_id] == caffe::ANNO_POSITIVE ||
							(anno_type[scale_base_id] == caffe::ANNO_IGNORE && loc_regress_on_ignore_)); // need loc_diff near the center of bbox
					bool need_bbox_ignore_generating = (anno_type[scale_base_id] == caffe::ANNO_IGNORE);
					if( ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) * ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) +
						((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal)) * ((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal)) > Dtype(1) )
					{
						if(dy != 0 || dx != 0){
							need_bbox_label_generating = false;
						}
					}
					if( ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) *
							((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) +
							((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale))) *
							((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale))) > Dtype(1) )
					{
						need_bbox_ignore_generating = false;
					}
					if( ((round(pt_y) - heat_map_y)/Dtype(max_loc_diff_valid_distance_vertical)) *
							((round(pt_y) - heat_map_y)/Dtype(max_loc_diff_valid_distance_vertical)) +
							((round(pt_x) - heat_map_x)/Dtype(max_loc_diff_valid_distance_horizontal)) *
							((round(pt_x) - heat_map_x)/Dtype(max_loc_diff_valid_distance_horizontal)) > Dtype(1) )
					{
						need_bbox_loc_generating = false;
					}
					if( false == need_bbox_label_generating && false == need_bbox_loc_generating && false == need_bbox_ignore_generating)
						continue;
					/**
					 *  generate ground truth
					 */
//					if(need_bbox_label_generating == true && need_bbox_loc_generating == false){
//						CHECK(false)<<"need_bbox_label_generating == true && need_bbox_ignore_generating == false ::"
//								<<" ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) * ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) +"
//						<<"((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal)) * ((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal)): "<<
//						 ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) * ((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical)) +
//												((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal)) * ((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal))<<
//												"((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) *"<<
//							"((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) +"<<
//							"((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale))) *"<<
//							"((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale))):"<<
//							((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) *
//														((round(pt_y) - heat_map_y)/Dtype(max_valid_distance_vertical+ (ignore_margin_ *scale))) +
//														((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale))) *
//														((round(pt_x) - heat_map_x)/Dtype(max_valid_distance_horizontal+ (ignore_margin_ *scale)));
//					}

					int map_size = prefetch_label.offset(0,1,0,0);
					if(need_bbox_label_generating)
					{
						int off_set = base_offset + (channel_detection_label_channel_offset_ + scale_base_id*total_class_num_ +class_id)*map_size;
						top_label[off_set] = 1;
					}
					if(need_bbox_ignore_generating)
					{
						int off_set = base_offset + (channel_detection_ignore_label_channel_offset_ + scale_base_id*total_class_num_ +class_id)*map_size;
						top_label[off_set] = 1;
					}
					if( need_bbox_loc_generating   )
					{
						Dtype heat_map_bb_lt_x = (coords_of_one_instance[idx1] - this->heat_map_b_)/this->heat_map_a_;
						Dtype heat_map_bb_lt_y = (coords_of_one_instance[idx1+1] - this->heat_map_b_)/this->heat_map_a_;
						Dtype heat_map_bb_rb_x = (coords_of_one_instance[idx2] - this->heat_map_b_)/this->heat_map_a_;
						Dtype heat_map_bb_rb_y = (coords_of_one_instance[idx2+1] - this->heat_map_b_)/this->heat_map_a_;
						int off_set = base_offset + (channel_detection_diff_channel_offset_ + 4 * (scale_base_id*total_class_num_ +class_id))*map_size;

						top_label[off_set] = 			  Dtype(round(pt_x) - heat_map_bb_lt_x);
						top_label[off_set + map_size] =   Dtype(round(pt_y) - heat_map_bb_lt_y);
						top_label[off_set + map_size*2] = Dtype(round(pt_x) - heat_map_bb_rb_x);
						top_label[off_set + map_size*3] = Dtype(round(pt_y) - heat_map_bb_rb_y);
					}
					if(need_bbox_loc_generating && need_point_diff_from_center_  )
					{
						int n_diff_from_center_points = this->point_id_for_point_diff_from_center_.size();

						for (int p_id_id = 0; p_id_id < n_diff_from_center_points;++p_id_id)
						{
							int p_id = point_id_for_point_diff_from_center_[p_id_id];
							int point_diff_offset = base_offset + (channel_point_diff_channel_offset_ +
																local_channel_point_diff_offset + p_id_id*2)*map_size;
							if(coords_of_one_instance[p_id*2]  != Dtype(-1) && coords_of_one_instance[p_id*2+1] != Dtype(-1) ){
								Dtype loc_x = (coords_of_one_instance[p_id*2] - this->heat_map_b_) / this->heat_map_a_;
								Dtype loc_y = (coords_of_one_instance[p_id*2+1] - this->heat_map_b_) / this->heat_map_a_;


								top_label[ point_diff_offset   ] = Dtype(round(pt_x) - loc_x);
								top_label[ point_diff_offset +  map_size ]  = Dtype(round(pt_y) - loc_y);

								int point_score_offset = base_offset + (channel_point_score_channel_offset_ +
										local_channel_point_score_offset + p_id_id)*map_size;
								if(this->has_point_ignore_flag_diff_from_center_){
									int ignore_p_id = this->point_ignore_flag_id_for_point_diff_from_center_[p_id_id];
									if(((coords_of_one_instance[ignore_p_id*2] == Dtype(1)) || coords_of_one_instance[ignore_p_id*2 + 1] == Dtype(1) )
											|| anno_type[scale_base_id] == caffe::ANNO_IGNORE){
										point_score_offset = base_offset + (channel_point_ignore_channel_offset_ +
												local_channel_point_score_offset + p_id_id)*map_size;
									}
								}else{
									point_score_offset = base_offset + (channel_point_score_channel_offset_ +
											local_channel_point_score_offset + p_id_id)*map_size;
								}
								top_label[ point_score_offset ] = Dtype(1);
							}
							// special case for when point anno is unavailable. just set the ignore flag for point diff score.
							if(this->has_point_ignore_flag_diff_from_center_){
								int ignore_p_id = this->point_ignore_flag_id_for_point_diff_from_center_[p_id_id];
								if(((coords_of_one_instance[ignore_p_id*2] == Dtype(1)) || coords_of_one_instance[ignore_p_id*2 + 1] == Dtype(1) )
													|| anno_type[scale_base_id] == caffe::ANNO_IGNORE){
									int point_score_offset = base_offset + (channel_point_ignore_channel_offset_ +
																					local_channel_point_score_offset + p_id_id)*map_size;
									top_label[ point_score_offset ] = Dtype(1);
								}
							}

						}
					}
				}
			}
		}
	}
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>::GenerateDetectionMap(
		int item_id,const vector<Dtype> & coords, Blob<Dtype>& prefetch_label,int used_scale_base_id)
{
	vector<Dtype> scales  = GetScalesOfAllInstances(coords);
	int n_scales = this->scale_base_.size();
	/**
	 * first draw others
	 */
	for(int instance_id = scales.size() -1; instance_id >= 0; -- instance_id)
	{
		if(this->out_height_ == 1 && this->out_width_ == 1 && instance_id != 0)
			continue;

		vector<Dtype> cur_coords (coords.begin() + instance_id * this->num_anno_points_per_instance_ *2,
				coords.begin() + (instance_id + 1) * this->num_anno_points_per_instance_ *2);
		vector<ImageDataAnnoType> anno_types = IImageDataProcessor<Dtype>::GetAnnoTypeForAllScaleBase(scales[instance_id]);

		// set anno_type to ignore if the current bbox has a ignore_class_flag
		if(this->ignore_class_flag_id_ != -1){
			if(ignore_class_flag_id_ < this->num_anno_points_per_instance_  ){
				for(int scale_id=0; scale_id < n_scales; ++scale_id){
					int cur_ignore_class_flag_1 = cur_coords[ this->ignore_class_flag_id_ * 2];
					int cur_ignore_class_flag_2 = cur_coords[ this->ignore_class_flag_id_ * 2 + 1];
					if(cur_ignore_class_flag_1 == cur_ignore_class_flag_2){
						if(cur_ignore_class_flag_1 == 1 && anno_types[scale_id] == caffe::ANNO_POSITIVE ){
							anno_types[scale_id] = caffe::ANNO_IGNORE;
						}
					}else{
						LOG(ERROR)<<"cur_ignore_class_flag_1 != cur_ignore_class_flag_2: "<<cur_ignore_class_flag_1<<
								"  "<<cur_ignore_class_flag_2;
					}
				}
			}else{
				LOG(ERROR)<<"ignore_class_flag_id_ is larger(equal) than num_anno_points_per_instance_: "<< ignore_class_flag_id_<<
						" "<< this->num_anno_points_per_instance_ ;
			}
		}
		GenerateDetectionMapForOneInstance(item_id, cur_coords,scales[instance_id], anno_types, prefetch_label,used_scale_base_id);
	}

}


template <typename Dtype>
bool IImageDataDetectionBox<Dtype>::IsLabelMapAllZero(const Blob<Dtype>& label, const int class_id,int item_id, int scale_base_id){
	int map_size = label.offset(0,1,0,0);
	const Dtype * top_label = label.cpu_data();
	for (int  h = 0; h < label.height(); ++h)
	{
		for(int w = 0 ; w < label.width(); ++ w)
		{
			int top_idx = label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);
			int label_base_offset  = top_idx + (channel_detection_label_channel_offset_+ scale_base_id*total_class_num_ +class_id) * map_size;
			if(top_label[label_base_offset ] != Dtype(0))
				return false;
		}
	}
	return true;
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>::LabelToVisualizedCVMat(const Blob<Dtype>& label, const int class_id,
		cv::Mat& out_probs, cv::Mat& ignore_out_probs,int item_id,
		int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
		bool need_regression, Dtype heat_map_a  , Dtype heat_map_b   )
{
	color_channel_weight[0] = MIN(1, MAX(0, color_channel_weight[0] ));
	color_channel_weight[1] = MIN(1, MAX(0, color_channel_weight[1] ));
	color_channel_weight[2] = MIN(1, MAX(0, color_channel_weight[2] ));
	int map_size = label.offset(0,1,0,0);
	const Dtype * top_label = label.cpu_data();
	for (int  h = 0; h < label.height(); ++h){
		for(int w = 0 ; w < label.width(); ++ w){
			int top_idx = label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);
			int label_base_offset  = top_idx + (channel_detection_label_channel_offset_+ scale_base_id*total_class_num_ +class_id) * map_size;
			int ignore_label_base_offset =top_idx + (channel_detection_ignore_label_channel_offset_ +  scale_base_id*total_class_num_ +class_id )* map_size;
			int detection_diff_base_offset = top_idx + (channel_detection_diff_channel_offset_ + (scale_base_id*total_class_num_ +class_id) *4 )* map_size;

			const int tmp_h = h * heat_map_a + heat_map_b;
			const int tmp_w = w * heat_map_a + heat_map_b;
			CHECK(this->CheckValidIndexInRange(out_probs, tmp_h,tmp_w));
			CHECK(this->CheckValidIndexInRange(ignore_out_probs, tmp_h,tmp_w));
			if(top_label[label_base_offset ] != Dtype(0)  && top_label[label_base_offset ] > threshold){
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = static_cast<uint8_t>(color_channel_weight[0] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(color_channel_weight[1] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = static_cast<uint8_t>(color_channel_weight[2] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));

				cv::circle( out_probs , cv::Point( tmp_w, tmp_h), 2, cv::Scalar(255, 255, 0));


			}

			if(top_label[ignore_label_base_offset] != Dtype(0)  && top_label[ignore_label_base_offset ] > threshold  ){
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(255 * MIN(1, MAX(0, top_label[ ignore_label_base_offset ])));
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = 0;
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 0;
				cv::circle( ignore_out_probs , cv::Point( tmp_w, tmp_h), 2, cv::Scalar(255, 255, 0));
			}

			if(this->need_detection_loc_diff_ && top_label[label_base_offset ] > threshold && need_regression){
				if(top_label[detection_diff_base_offset]  != Dtype(0)){
					Dtype coord[4];
					coord[0] = tmp_w- top_label[detection_diff_base_offset] * heat_map_a ;
					coord[0] = MIN(out_probs.cols, MAX(0, coord[0]));
					coord[1] = tmp_h- top_label[detection_diff_base_offset+ map_size*1]* heat_map_a ;
					coord[1] = MIN(out_probs.rows, MAX(0, coord[1]));
					coord[2] = tmp_w- top_label[detection_diff_base_offset+ map_size*2]* heat_map_a ;
					coord[2] = MIN(out_probs.cols, MAX(0, coord[2]));
					coord[3] = tmp_h- top_label[detection_diff_base_offset+ map_size*3]* heat_map_a ;
					coord[3] = MIN(out_probs.rows, MAX(0, coord[3]));
					cv::rectangle(out_probs, cv::Point( coord[0], coord[1]),
										cv::Point(coord[2],coord[3]), cv::Scalar(255 * color_channel_weight[0] ,
												255 * color_channel_weight[1] , 255 * color_channel_weight[2] ));
					out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 255;
				}

			}
		}
	}
}


template <typename Dtype>
void IImageDataDetectionBox<Dtype>::PointLabelToVisualizedCVMat(const Blob<Dtype>& label, const int class_id,
		int point_id, cv::Mat& out_probs, cv::Mat& ignore_out_probs,int item_id,
		int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
		bool need_regression, Dtype heat_map_a  , Dtype heat_map_b   )
{
	color_channel_weight[0] = MIN(1, MAX(0, color_channel_weight[0] ));
	color_channel_weight[1] = MIN(1, MAX(0, color_channel_weight[1] ));
	color_channel_weight[2] = MIN(1, MAX(0, color_channel_weight[2] ));
	int map_size = label.offset(0,1,0,0);
	const Dtype * top_label = label.cpu_data();
	int n_diff_from_center_points = this->point_id_for_point_diff_from_center_.size();


	int local_channel_point_score_offset =  n_diff_from_center_points *  (scale_base_id*total_class_num_ +class_id);
	int local_channel_point_diff_offset =  n_diff_from_center_points *  (scale_base_id*total_class_num_ +class_id)*2;

	for (int  h = 0; h < label.height(); ++h){
		for(int w = 0 ; w < label.width(); ++ w){
			int top_idx = label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);

			int label_score_offset = top_idx +  (channel_point_score_channel_offset_ +
					local_channel_point_score_offset + point_id)*map_size;
			const int tmp_h = h * heat_map_a + heat_map_b;
			const int tmp_w = w * heat_map_a + heat_map_b;
			CHECK(this->CheckValidIndexInRange(out_probs, tmp_h,tmp_w));
			CHECK(this->CheckValidIndexInRange(ignore_out_probs, tmp_h,tmp_w));
			bool draw_circle_on_label = false;
			if(top_label[label_score_offset ] != Dtype(0)  && top_label[label_score_offset ] > threshold){
				draw_circle_on_label = true;
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = static_cast<uint8_t>(color_channel_weight[0] * 255 * MIN(1, MAX(0, top_label[ label_score_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(color_channel_weight[1] * 255 * MIN(1, MAX(0, top_label[ label_score_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = static_cast<uint8_t>(color_channel_weight[2] * 255 * MIN(1, MAX(0, top_label[ label_score_offset ])));
			}
			bool draw_circle_on_ignore = false;
			if(this->has_point_ignore_flag_diff_from_center_){
				int ignore_label_base_offset  = top_idx +  (channel_point_ignore_channel_offset_ +
						local_channel_point_score_offset + point_id)*map_size;
				if(top_label[ignore_label_base_offset] != Dtype(0)  && top_label[ignore_label_base_offset ] > threshold  ){
					draw_circle_on_ignore = true;
					ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(255 * MIN(1, MAX(0, top_label[ ignore_label_base_offset ])));
					ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = 0;
					ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 0;
				}
			}
			int diff_from_center_offset = top_idx +  (channel_point_diff_channel_offset_ +
					local_channel_point_diff_offset + point_id*2 )*map_size;
			Dtype diff_x = top_label[diff_from_center_offset  ];
			Dtype diff_y = top_label[diff_from_center_offset +  map_size ];
			if ( diff_x != Dtype(0) && diff_y != Dtype(0) && (draw_circle_on_label || draw_circle_on_ignore)){

				Dtype coord[2];
				coord[0] = tmp_w- diff_x * heat_map_a ;
				coord[0] = MIN(out_probs.cols, MAX(0, coord[0]));
				coord[1] = tmp_h- diff_y * heat_map_a;
				coord[1] = MIN(out_probs.rows, MAX(0, coord[1]));
				cv::circle( out_probs ,
						cv::Point( coord[0], coord[1]), 2, cv::Scalar(255, 255, 0));
				if(draw_circle_on_ignore){
					cv::circle( ignore_out_probs ,
									cv::Point( coord[0], coord[1]), 2, cv::Scalar(255, 255, 0));
				}
			}
		}
	}
}



template <typename Dtype>
void IImageDataDetectionBox<Dtype>:: PrintPic(int item_id, const string & output_path,
		cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label)
{

	cv::Mat & cv_img = *( (img_cv_ptr));

	string out_sample_name = IImageDataProcessor<Dtype>::GetPrintSampleName(cur_sample,sample_type);
	char path[512];
	cv::Size output_size(this->input_width_, this->input_height_);
	LOG(INFO)<<"channel_detection_base_channel_offset_:"<<channel_detection_base_channel_offset_<<" "<<
			"channel_detection_label_channel_offset_:"<<channel_detection_label_channel_offset_<<" "<<
			"channel_detection_ignore_label_channel_offset_:"<<channel_detection_ignore_label_channel_offset_<<" "<<
			"channel_detection_diff_channel_offset_:"<<channel_detection_diff_channel_offset_<<" "<<
			"channel_point_diff_from_center_channel_offset_:"<<channel_point_diff_from_center_channel_offset_;

	Dtype color_channel_weight[3] = {0,1,0};
	for(int scale_base_id = 0 ; scale_base_id < this->scale_base_.size(); ++ scale_base_id)
	{
		for(int class_id = 0; class_id < this->total_class_num_; ++class_id){

//			bool ignore_this_class = IsLabelMapAllZero(prefetch_label,  class_id, item_id,  scale_base_id);
//			if(ignore_this_class){
//				LOG(INFO)<<"Ignore class_id "<<class_id <<" in PrintPic";
//				continue;
//			}

			cv::Mat out_probs, ignore_out_probs;
			cv::resize(cv_img, out_probs,output_size);
			cv::resize(cv_img, ignore_out_probs,output_size);

			LabelToVisualizedCVMat(prefetch_label, class_id,out_probs, ignore_out_probs,item_id, scale_base_id,
					color_channel_weight, -1, true,this->heat_map_a_, this->heat_map_b_);

			sprintf(path, "%s/pic/det_%03d_class_%03d_%s_s%f_s%f_bbox_label_scalebase.jpg", output_path.c_str(),item_id,class_id,
					out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			LOG(INFO)<< "Saving gaussian map: " << path;
			imwrite(path, out_probs);
			sprintf(path, "%s/pic/det_%03d_class_%03d_%s_s%f_s%f_bbox_ignore_scalebase.jpg", output_path.c_str(),item_id,class_id,
							out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			LOG(INFO)<< "Saving gaussian map: " << path;
			imwrite(path, ignore_out_probs);

			if(this->need_point_diff_from_center_){
				for(int p_id = 0; p_id < this->point_id_for_point_diff_from_center_.size();++p_id){

					cv::resize(cv_img, out_probs,output_size);
					cv::resize(cv_img, ignore_out_probs,output_size);
					PointLabelToVisualizedCVMat(prefetch_label, class_id, p_id,out_probs, ignore_out_probs,item_id, scale_base_id,
							color_channel_weight, -1, true,this->heat_map_a_, this->heat_map_b_);

					sprintf(path, "%s/pic/det_point%03d_%03d_class_%03d_%s_s%f_s%f_bbox_label_scalebase.jpg", output_path.c_str(),
							p_id,item_id,class_id, out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
					LOG(INFO)<< "Saving gaussian map: " << path;
					imwrite(path, out_probs);
					sprintf(path, "%s/pic/det_point%03d_%03d_class_%03d_%s_s%f_s%f_bbox_ignore_scalebase.jpg", output_path.c_str(),
							p_id,item_id,class_id, out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
					LOG(INFO)<< "Saving gaussian map: " << path;
					imwrite(path, ignore_out_probs);
				}
			}
		}
	}
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>:: PrintLabel(int item_id, const string & output_path,
		const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label)
{
	const Dtype * top_label = prefetch_label.cpu_data();
	string out_sample_name = IImageDataProcessor<Dtype>::GetPrintSampleName(cur_sample,sample_type);

	char path[512];
	int map_size = prefetch_label.offset(0,1,0,0);
	for(int scale_base_id = 0 ; scale_base_id < this->scale_base_.size(); ++ scale_base_id)
	{
		for(int class_id = 0; class_id < total_class_num_; ++class_id){
			sprintf(path, "%s/label/%03d_class_%03d_%s_s%f_s%f_bbox_label_scalebase.txt", output_path.c_str(),item_id,class_id,
								out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			std::ofstream label_file(path);
			sprintf(path, "%s/label/%03d_class_%03d_%s_s%f_s%f_bbox_ignore_scalebase.txt", output_path.c_str(),item_id,class_id,
										out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			std::ofstream ignore_file(path);
			/**
			 * for label and ingore region
			 */
			for (int h = 0; h < prefetch_label.height(); ++h) {
				for (int w = 0; w < prefetch_label.width(); ++w) {
					int top_idx = prefetch_label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);
					int label_base_offset  = top_idx + (channel_detection_label_channel_offset_+scale_base_id*total_class_num_ +class_id) * map_size;
					int ignore_label_base_offset =top_idx + (channel_detection_ignore_label_channel_offset_ + scale_base_id*total_class_num_ +class_id )* map_size;
					label_file <<std::setiosflags(ios::fixed)<< setprecision(0)<< top_label[label_base_offset] << " ";
					ignore_file <<std::setiosflags(ios::fixed)<< setprecision(0)<< top_label[ignore_label_base_offset] << " ";
				}
				label_file << "\n";
				ignore_file << "\n";
			}
			label_file.close();
			ignore_file.close();
			LOG(INFO)<< "Saving label_file: " << path;

			/**
			 * for bbox regression (dx, dy)
			 */
			for(int p_id = 0 ; p_id< 4; ++p_id)
			{
				sprintf(path, "%s/label/%03d_%s_class_%03d_bbox_diff_%d_%f__base_scale_%f.txt",
						 output_path.c_str(),item_id, out_sample_name.c_str(),class_id,p_id, float(scale),float(this->scale_base_[scale_base_id]));
				std::ofstream diff_file(path);
				for (int h = 0; h < prefetch_label.height(); ++h) {
					for (int w = 0; w < prefetch_label.width(); ++w) {
						int top_idx = prefetch_label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);
						int detection_diff_base_offset = top_idx + (channel_detection_diff_channel_offset_ + (scale_base_id*total_class_num_ +class_id) *4 )* map_size;
						diff_file <<std::setiosflags(ios::fixed)<< setprecision(2)<<
								top_label[detection_diff_base_offset+ map_size*p_id] << " ";
					}
					diff_file << "\n";
				}
				diff_file.close();
			}
			/**
			 * for point_diff_from_center
			 */
			for(int p_id = 0 ; p_id< point_id_for_point_diff_from_center_.size(); ++p_id)
			{
				sprintf(path, "%s/label/%03d_%s_class_%03d_point_diff_from_center_x_%d_%f__base_scale_%f.txt",
						 output_path.c_str(),item_id, out_sample_name.c_str(),class_id,p_id, float(scale),float(this->scale_base_[scale_base_id]));
				std::ofstream diff_x_file(path);
				sprintf(path, "%s/label/%03d_%s_class_%03d_point_diff_from_center_y_%d_%f__base_scale_%f.txt",
						 output_path.c_str(), item_id,out_sample_name.c_str(),class_id,p_id, float(scale),float(this->scale_base_[scale_base_id]));
				std::ofstream diff_y_file(path);

				for (int h = 0; h < prefetch_label.height(); ++h) {
					for (int w = 0; w < prefetch_label.width(); ++w) {
						int top_idx = prefetch_label.offset(item_id, this->channel_detection_base_channel_offset_, h , w);
						int point_diff_from_center_channel_offset = top_idx + (channel_point_diff_from_center_channel_offset_ +
												point_id_for_point_diff_from_center_.size() * 2 * (scale_base_id*total_class_num_ +class_id)) * map_size;
						diff_x_file <<std::setiosflags(ios::fixed)<< setprecision(2)<<
								top_label[point_diff_from_center_channel_offset + (p_id*2)*map_size] << " ";
						diff_y_file <<std::setiosflags(ios::fixed)<< setprecision(2)<<
													top_label[point_diff_from_center_channel_offset + (p_id*2 + 1)*map_size] << " ";
					}
					diff_x_file << "\n";
					diff_y_file << "\n";
				}
				diff_x_file.close();
				diff_y_file.close();
			}
		}
	}
}

template <typename Dtype>
void IImageDataDetectionBox<Dtype>:: ShowDataAndPredictedLabel(const string & output_path,const string & img_name,
		const Blob<Dtype>& data, const int sampleID,const Dtype* mean_bgr,const Blob<Dtype>& label
		,const Blob<Dtype>& predicted,Dtype threshold)
{
	cv::Size label_size(label.width(), label.height());
	Dtype color_channel_weight[3] = {0,1,0};
	char path[512];

	vector<int> params_jpg;
	params_jpg.push_back(CV_IMWRITE_JPEG_QUALITY);
	params_jpg.push_back(100);

	int nums = data.num();
	CHECK_EQ(nums, label.num());


	for(int scale_base_id = 0 ; scale_base_id < this->scale_base_.size(); ++ scale_base_id)
	{
		for(int class_id = 0; class_id < this->total_class_num_; ++class_id){
			cv::Mat cv_img_original = caffe::BlobImgDataToCVMat(data, sampleID, mean_bgr[0],
					mean_bgr[1],mean_bgr[2]);
			cv::Mat original_clone, original_clone_ignore ;
			cv::Mat cv_img_original_mix;

			if(this->need_point_diff_from_center_){
				for(int p_id=0; p_id < this->point_id_for_point_diff_from_center_.size();++p_id){
					cv::Mat cv_img_original_gt_pred(cv_img_original.rows, cv_img_original.cols * 2 + this->PIC_MARGIN, CV_8UC3);


					cv::Mat cv_img_gt = cv::Mat(label.height(), label.width(), CV_8UC3);
					cv_img_gt = cv::Mat::zeros(label.height(), label.width(), CV_8UC3);
					cv::Mat cv_img_gt_ignore = cv_img_gt.clone();
					cv::Mat cv_img_pred = cv_img_gt.clone();


					color_channel_weight[0] = 1;
					color_channel_weight[1] = 1;
					color_channel_weight[2] = 1;
					PointLabelToVisualizedCVMat(label,class_id,p_id, cv_img_gt, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, -1,false );
					original_clone = cv_img_original.clone();
					original_clone_ignore = cv_img_original.clone();
					/**
					 * the ground truth information is stored in the second channel
					 */
					color_channel_weight[0] = 0;
					color_channel_weight[1] = 1;
					color_channel_weight[2] = 0;
					PointLabelToVisualizedCVMat(label, class_id,p_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, -1,
							true,this->heat_map_a_, this->heat_map_b_ );

					original_clone.copyTo(cv_img_original_gt_pred(
							cv::Rect(0,0,original_clone.cols, original_clone.rows)));

					color_channel_weight[0] = 1;
					color_channel_weight[1] = 1;
					color_channel_weight[2] = 1;
					PointLabelToVisualizedCVMat(predicted,class_id, p_id,cv_img_pred, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, threshold,false );

					original_clone = cv_img_original.clone();
					/**
					 * the prediction information is stored in the first channel
					 */
					color_channel_weight[0] = 1;
					color_channel_weight[1] = 0;
					color_channel_weight[2] = 0;
					PointLabelToVisualizedCVMat(predicted, class_id,p_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, threshold,
							true,this->heat_map_a_, this->heat_map_b_ );

					original_clone.copyTo(cv_img_original_gt_pred(cv::Rect(
							original_clone.cols + this->PIC_MARGIN, 0,
							original_clone.cols, original_clone.rows)));

					sprintf(path, "%s/%s_point_%03d_bbox_original_gt_pred_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str(),p_id,class_id,scale_base_id );
					imwrite(path, cv_img_original_gt_pred, params_jpg);
					sprintf(path, "%s/%s_point_%03d_bbox_gt_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,p_id,class_id,scale_base_id);
					imwrite(path, cv_img_gt, params_jpg);
					sprintf(path, "%s/%s_point_%03d_bbox_pred_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,p_id,class_id,scale_base_id);
					imwrite(path, cv_img_pred, params_jpg);

				}
			}


			cv::resize(cv_img_original, cv_img_original_mix, label_size);
			cv::Mat cv_img_original_gt_pred(cv_img_original.rows, cv_img_original.cols * 2 + this->PIC_MARGIN, CV_8UC3);


			cv::Mat cv_img_gt = cv::Mat(label.height(), label.width(), CV_8UC3);
			cv_img_gt = cv::Mat::zeros(label.height(), label.width(), CV_8UC3);
			cv::Mat cv_img_gt_ignore = cv_img_gt.clone();
			cv::Mat cv_img_pred = cv_img_gt.clone();

			color_channel_weight[0] = 1;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 1;
			LabelToVisualizedCVMat(label,class_id, cv_img_gt, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, -1,false );
			original_clone = cv_img_original.clone();
			original_clone_ignore = cv_img_original.clone();
			/**
			 * the ground truth information is stored in the second channel
			 */
			color_channel_weight[0] = 0;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 0;
			LabelToVisualizedCVMat(label, class_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, -1,
					true,this->heat_map_a_, this->heat_map_b_ );
			LabelToVisualizedCVMat(label, class_id,cv_img_original_mix, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, -1,false );
			original_clone.copyTo(cv_img_original_gt_pred(
					cv::Rect(0,0,original_clone.cols, original_clone.rows)));

			color_channel_weight[0] = 1;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 1;
			LabelToVisualizedCVMat(predicted,class_id, cv_img_pred, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, threshold,false );

			original_clone = cv_img_original.clone();
			/**
			 * the prediction information is stored in the first channel
			 */
			color_channel_weight[0] = 1;
			color_channel_weight[1] = 0;
			color_channel_weight[2] = 0;
			LabelToVisualizedCVMat(predicted, class_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, threshold,
					true,this->heat_map_a_, this->heat_map_b_ );
			LabelToVisualizedCVMat(predicted, class_id,cv_img_original_mix, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, threshold,false );
			original_clone.copyTo(cv_img_original_gt_pred(cv::Rect(
					original_clone.cols + this->PIC_MARGIN, 0,
					original_clone.cols, original_clone.rows)));

			sprintf(path, "%s/%s_bbox_original_gt_pred_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str(),class_id,scale_base_id );
			imwrite(path, cv_img_original_gt_pred, params_jpg);
			sprintf(path, "%s/%s_bbox_gt_pred_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,class_id,scale_base_id);
			imwrite(path, cv_img_original_mix, params_jpg);
			sprintf(path, "%s/%s_bbox_gt_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,class_id,scale_base_id);
			imwrite(path, cv_img_gt, params_jpg);
			sprintf(path, "%s/%s_bbox_pred_class_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,class_id,scale_base_id);
			imwrite(path, cv_img_pred, params_jpg);
		}
	}

}

INSTANTIATE_CLASS(IImageDataDetectionBox);
}  // namespace caffe

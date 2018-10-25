#include <string>
#include <vector>

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
namespace caffe {

template <typename Dtype>
IImageDataKeyPoint<Dtype>::IImageDataKeyPoint(){
	channel_point_base_offset_ = 0;
	channel_point_loc_diff_offset_ = 0;
	channel_point_ignore_point_channel_offset_ = 0;
	channel_point_channel_all_need_ = 0;
	channel_point_valid_keypoint_channel_offset_ = 0;
	key_point_valid_distance_ = 0;
	key_point_min_out_valid_len_ = 0;
	ignore_flag_radius_ = 0;
	need_point_loc_diff_ = false;
	key_point_loc_diff_radius_ = 0;
	used_key_point_idxs_.clear();
	ignore_key_point_flag_idxs_.clear();
	key_point_valid_flag_.clear();
	get_ignore_flag_idx_by_point_idx_.clear();
	used_attribute_point_idxs_.clear();
	attribute_point_flag_idxs_.clear();
	key_point_standard_length_ = 0;
	key_point_standard_point_id1_ = 0;
	key_point_standard_point_id2_ = 0;
	used_key_point_channel_offset_.clear();
	object_center_point_id_  = -1;
}

template <typename Dtype>
IImageDataKeyPoint<Dtype>::~IImageDataKeyPoint(){

}


template <typename Dtype>
void IImageDataKeyPoint<Dtype>::SetUpParameter(
		const FCNImageDataParameter& fcn_image_data_param)
{
	IImageDataProcessor<Dtype>::SetUpParameter(fcn_image_data_param);
	CHECK(fcn_image_data_param.has_fcn_image_data_key_point_param())<<
		"fcn_image_data_key_point_param is needed";
	this->SetUpParameter(fcn_image_data_param.fcn_image_data_key_point_param());
}

/**
 * For all scales, annotation: point 1,... point n, ignore_point1 , ..., ignore_pointn,.... dx1, dy1, ......  dxn, dyn
 *
 * Note: call SetUpParameter first before calling SetUpChannelInfo;
 */
template <typename Dtype>
int IImageDataKeyPoint<Dtype>::SetUpChannelInfo( const int channel_base_offset )
{
	int n_scale_base = this->scale_base_.size();
	channel_point_base_offset_ = channel_base_offset;
	channel_point_ignore_point_channel_offset_ = n_scale_base * used_key_point_idxs_.size();
	channel_point_loc_diff_offset_ = channel_point_ignore_point_channel_offset_ + n_scale_base * used_key_point_idxs_.size();

	channel_point_attribute_point_channel_offset_ = channel_point_loc_diff_offset_ +
			n_scale_base * used_key_point_idxs_.size()*2;

	channel_point_channel_all_need_ = channel_point_attribute_point_channel_offset_ +
			n_scale_base * this->used_attribute_point_idxs_.size();

	this->total_channel_need_ = channel_point_base_offset_ + channel_point_channel_all_need_;

	LOG(INFO)<<"total number of label channels for key points: "<<channel_point_channel_all_need_<<
			" in which valid key point need "<< channel_point_ignore_point_channel_offset_ - channel_point_valid_keypoint_channel_offset_<<
			" , and ignore point need "<< channel_point_loc_diff_offset_ - channel_point_ignore_point_channel_offset_<<
			" and loc_diff for valid points need "<< channel_point_attribute_point_channel_offset_ - channel_point_loc_diff_offset_<<
			" and attribute_point_channel_ need "<< channel_point_channel_all_need_ - channel_point_attribute_point_channel_offset_;

	return channel_point_base_offset_ + channel_point_channel_all_need_;

}
template <typename Dtype>
void IImageDataKeyPoint<Dtype>::SetUpParameter(
		const FCNImageDataKeyPointParameter& fcn_img_data_keypoint_param)
{

	this->key_point_valid_distance_  = fcn_img_data_keypoint_param.key_point_valid_distance();
	this->key_point_min_out_valid_len_ = fcn_img_data_keypoint_param.key_point_min_out_valid_len();

	CHECK(fcn_img_data_keypoint_param.has_used_key_points_file());
	/*
	 * Read Keypoint
	 * */
	this->key_point_valid_flag_.resize(this->num_anno_points_per_instance_,false);
	std::ifstream key_points_file(fcn_img_data_keypoint_param.used_key_points_file().c_str());
	CHECK(key_points_file.good()) << "Failed to open used key point file "
			<< fcn_img_data_keypoint_param.used_key_points_file() << std::endl;
	this->used_key_point_idxs_.clear();

	std::ostringstream oss;
	int key_point;
	used_key_point_channel_offset_.resize(this->num_anno_points_per_instance_,-1);
	while(key_points_file >> key_point) {
	  CHECK_LT(key_point,this->num_anno_points_per_instance_)<<"Key point id should not exceed the "<<
			  "number of point per instance";
	  key_point_valid_flag_[key_point] = true;
	  used_key_point_channel_offset_[key_point] = used_key_point_idxs_.size();
	  this->used_key_point_idxs_.push_back(key_point);
	  oss << key_point << " ";
	}
	key_points_file.close();
	LOG(INFO) << "There are " << used_key_point_idxs_.size()
	<< " used key points: " << oss.str();

	/*
	 * Read keypoint flags for ignore label
	 */

	this->ignore_flag_radius_ = fcn_img_data_keypoint_param.ignore_flag_radius();
	get_ignore_flag_idx_by_point_idx_.resize(this->num_anno_points_per_instance_,-1);

	if(fcn_img_data_keypoint_param.has_ignore_flag_list_file()) {
		std::ifstream ignore_flag_file(fcn_img_data_keypoint_param.ignore_flag_list_file().c_str());
		CHECK(ignore_flag_file.good()) << "Failed to open key point flag list file "
		<< fcn_img_data_keypoint_param.ignore_flag_list_file() << std::endl;
		std::ostringstream oss_ignored_point;
		int ignore_key_point;

		while(ignore_flag_file >> ignore_key_point) {
			CHECK_LT(ignore_key_point,this->num_anno_points_per_instance_)<<"Ignore key point id should not exceed the "<<
						  "number of point per instance";
			int corresponding_valid_key_point_idx =used_key_point_idxs_[ignore_key_point_flag_idxs_.size()];
			get_ignore_flag_idx_by_point_idx_[corresponding_valid_key_point_idx] = ignore_key_point;
			ignore_key_point_flag_idxs_.push_back(ignore_key_point);
			oss_ignored_point << ignore_key_point << " ";

		}
		ignore_flag_file.close();

		LOG(INFO) << "There are " << ignore_key_point_flag_idxs_.size()
			<< " ignore flag  points: " << oss_ignored_point.str();
		CHECK_EQ(ignore_key_point_flag_idxs_.size(),this->used_key_point_idxs_.size())<<"The ignore key points and "<<
			"used key points should have the same size.";
	}

	if(fcn_img_data_keypoint_param.has_attribute_point_files()){
		get_attribute_idx_by_point_idx_.resize(this->num_anno_points_per_instance_, false);
		attribute_point_flag_idxs_.resize(this->num_anno_points_per_instance_, false);
		used_attribute_point_channel_offset_.resize(this->num_anno_points_per_instance_,-1);
		std::ifstream attribute_file(fcn_img_data_keypoint_param.attribute_point_files().c_str());
		CHECK(attribute_file.good()) << "Failed to open attribute point file "
				<< fcn_img_data_keypoint_param.attribute_point_files() << std::endl;
		std::ostringstream oss_attribute_point;
		int attribute_point;
		while(attribute_file >> attribute_point){
			CHECK_LT(attribute_point,this->num_anno_points_per_instance_)<<"Attribute point id should not exceed the "<<
									  "number of point per instance";
			this->attribute_point_flag_idxs_[attribute_point] = true;
			used_attribute_point_channel_offset_[attribute_point] = used_attribute_point_idxs_.size();
			int corresponding_valid_key_point_idx =used_key_point_idxs_[used_attribute_point_idxs_.size()];
			get_attribute_idx_by_point_idx_[corresponding_valid_key_point_idx] = attribute_point;

			used_attribute_point_idxs_.push_back(attribute_point);
			oss_attribute_point << attribute_point << " ";
		}
		attribute_file.close();
		LOG(INFO) << "The attribute point idx: "  << oss_attribute_point.str();
		CHECK_EQ(used_attribute_point_idxs_.size(),this->used_key_point_idxs_.size())<<"The attribute points and "<<
					"used key points should have the same size.";
	}

	/*
	 * Setup if location distance from key points is needed
	 */
	this->need_point_loc_diff_ = fcn_img_data_keypoint_param.need_point_loc_diff();
	this->key_point_loc_diff_radius_ = fcn_img_data_keypoint_param.key_point_loc_diff_radius();

	CHECK(fcn_img_data_keypoint_param.has_key_point_standard_point_id1());
	this->key_point_standard_point_id1_ = fcn_img_data_keypoint_param.key_point_standard_point_id1();
	CHECK_LT(key_point_standard_point_id1_,this->num_anno_points_per_instance_);
	CHECK(fcn_img_data_keypoint_param.has_key_point_standard_point_id2());
	this->key_point_standard_point_id2_ = fcn_img_data_keypoint_param.key_point_standard_point_id2();
	CHECK_LT(key_point_standard_point_id2_,this->num_anno_points_per_instance_);
	CHECK(fcn_img_data_keypoint_param.has_key_point_standard_length());
	this->key_point_standard_length_ = fcn_img_data_keypoint_param.key_point_standard_length();

	if(fcn_img_data_keypoint_param.has_object_center_point_id()){
		object_center_point_id_ = fcn_img_data_keypoint_param.object_center_point_id();
		CHECK_LT(object_center_point_id_,this->num_anno_points_per_instance_);
	}

}


template <typename Dtype>
Dtype IImageDataKeyPoint<Dtype>::GetScaleForCurInstance(const vector<Dtype> & coords_of_one_instance){
	Dtype x1 = coords_of_one_instance[this->key_point_standard_point_id1_*2];
	Dtype y1 = coords_of_one_instance[this->key_point_standard_point_id1_*2+1];
	Dtype x2 = coords_of_one_instance[this->key_point_standard_point_id2_*2];
	Dtype y2 = coords_of_one_instance[this->key_point_standard_point_id2_*2+1];
	if( x1 == -1 || y1 == -1 || x2 == -1 || y2 == -1){
		return 1;
	}
	Dtype dist = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
	if(dist < 0.1){
		LOG(INFO)<<"Warning! The distance calculated in IImageDataKeyPoint<Dtype>::GetScaleForCurInstance is too small"
				<< dist;
	}
	return dist/this->key_point_standard_length_;

}

template <typename Dtype>
vector<Dtype> IImageDataKeyPoint<Dtype>::GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance){
	vector<Dtype> scales ;
	int num_instances =  coords_of_all_instance.size() / (this->num_anno_points_per_instance_ * 2);
	CHECK(coords_of_all_instance.size() % (this->num_anno_points_per_instance_ * 2) == 0);
	CHECK_GT(num_instances,0);

	for(int i = 0; i < num_instances; ++i ){
		int cur_point_idx1 = (key_point_standard_point_id1_ + this->num_anno_points_per_instance_  * i )*2;
		int cur_point_idx2 = (key_point_standard_point_id2_ + this->num_anno_points_per_instance_  * i )*2;
		Dtype dx = coords_of_all_instance[cur_point_idx2] - coords_of_all_instance[cur_point_idx1];
		Dtype dy = coords_of_all_instance[cur_point_idx2+1] - coords_of_all_instance[cur_point_idx1+1];
		Dtype len = sqrt(dx * dx + dy * dy);
		scales.push_back(len / this->key_point_standard_length_);
	}
	return scales;
}


template <typename Dtype>
void IImageDataKeyPoint<Dtype>::FilterAnnoTypeByObjCenter(vector<ImageDataAnnoType>& anno_type,
		vector<Dtype>& cur_coords)
{
	if(this->object_center_point_id_ >= 0){
		Dtype coord_x = cur_coords[object_center_point_id_*2];
		Dtype coord_y = cur_coords[object_center_point_id_*2+1];
		bool out_of_img =  coord_x <= 0 || coord_y <= 0 || coord_x >= this->input_width_ || coord_y >= this->input_height_;
		if(out_of_img){
			for(int i=0; i < anno_type.size(); ++i){
				anno_type[i] = anno_type[i] == caffe::ANNO_POSITIVE ? caffe::ANNO_IGNORE : anno_type[i];
			}
		}
	}
}

template <typename Dtype>
void IImageDataKeyPoint<Dtype>::GenerateKeyPointHeatMap(
		int item_id, vector<Dtype> & coords,  Blob<Dtype>& prefetch_label,int used_scale_base_id)
{
	vector<Dtype> scales  = GetScalesOfAllInstances(coords);

	for(int instance_id = 0; instance_id < scales.size(); ++ instance_id){
		vector<Dtype> cur_coords (coords.begin() + instance_id * this->num_anno_points_per_instance_ *2,
						coords.begin() + (instance_id + 1) * this->num_anno_points_per_instance_ *2);
		vector<ImageDataAnnoType> anno_types = this->GetAnnoTypeForAllScaleBase(scales[instance_id]);

//		for(int i=0; i < cur_coords.size()/2; ++i){
//			LOG(INFO)<<"coord["<<i<<"] = ["<<cur_coords[i*2]<<","<<cur_coords[i*2+1]<<"]";
//		}

		FilterAnnoTypeByObjCenter( anno_types,  cur_coords);
		GenerateKeyPointHeatMapForOneInstance(item_id, cur_coords,scales[instance_id], anno_types, prefetch_label);
	}
}


template <typename Dtype>
void IImageDataKeyPoint<Dtype>::GenerateKeyPointHeatMapForOneInstance(
		int item_id, const vector<Dtype> & coords_of_one_instance,
		  const Dtype scale, const vector<ImageDataAnnoType> anno_type,Blob<Dtype>& prefetch_label)
{
	CHECK_GE(coords_of_one_instance.size(), this->num_anno_points_per_instance_ *2);
	Dtype* top_label =  prefetch_label.mutable_cpu_data();
	int scale_base_size = this->scale_base_.size();
	int map_size = prefetch_label.offset(0,1,0,0);

	/**
	 * loop for each scale_base
	 */
	for(int scale_base_id = 0 ; scale_base_id < scale_base_size; ++ scale_base_id){
		if(anno_type[scale_base_id] == caffe::ANNO_NEGATIVE){
			continue;
		}
		/**
		 * for each point
		 */
		for(int idx = 0; idx < this->num_anno_points_per_instance_; ++idx){
			if(this->key_point_valid_flag_[idx] == false){
				continue;
			}
			int corresponding_ignore_point_flag_idx = get_ignore_flag_idx_by_point_idx_[idx];
//			LOG(INFO)<<"corresponding_ignore_point_flag_idx: "<<corresponding_ignore_point_flag_idx;
//			for(int i=0; i < get_ignore_flag_idx_by_point_idx_.size(); ++i){
//				LOG(INFO)<<"get_ignore_flag_idx_by_point_idx_["<<i<<"] = "<<get_ignore_flag_idx_by_point_idx_[i];
//			}

			bool cur_point_ignore = false;
			if(ignore_key_point_flag_idxs_.size() > 0){
				CHECK_GE(corresponding_ignore_point_flag_idx,0)<<"Error, get_ignore_flag_idx_by_point_idx_["<< idx<<"] == -1";
				cur_point_ignore = (coords_of_one_instance[corresponding_ignore_point_flag_idx*2] == 1);
			}



			bool cur_has_attribute = false;
			int corresponding_attribute_idx = -1;
			if(this->used_attribute_point_idxs_.size() > 0){
				corresponding_attribute_idx = this->get_attribute_idx_by_point_idx_[idx];
				if(this->used_attribute_point_idxs_.size() > 0 && corresponding_attribute_idx >= 0){
					cur_has_attribute = true;
				}
			}

			int point_idx = idx;
			int idx1 = point_idx*2;
			int idx2 = point_idx*2+1;
			if(coords_of_one_instance[idx1] != -1 && coords_of_one_instance[idx2] != -1){
				Dtype heat_map_x = (coords_of_one_instance[idx1] - this->heat_map_b_)/this->heat_map_a_;
				Dtype heat_map_y = (coords_of_one_instance[idx2] - this->heat_map_b_)/this->heat_map_a_;
//				LOG(INFO)<<"for point_id "<<idx<<" , coords_of_one_instance[idx1]="<<coords_of_one_instance[idx1]
//				            <<" coords_of_one_instance[idx2] ="<<coords_of_one_instance[idx2] ;
//				LOG(INFO)<<"for point_id "<<idx<<" , heat_map_x="<<heat_map_x<<" heat_map_y="<<heat_map_y;
				int key_point_valid_len = this->key_point_min_out_valid_len_;
				int cur_valid_point_channel_offset = scale_base_id * used_key_point_idxs_.size() + used_key_point_channel_offset_[idx];
				int cur_diff_channel_offset = channel_point_loc_diff_offset_ + (cur_valid_point_channel_offset)*2;
				int cur_attribute_channel_offset = -1;
				if(cur_has_attribute){
					cur_attribute_channel_offset =  channel_point_attribute_point_channel_offset_ +
						scale_base_id * this->used_attribute_point_idxs_.size() + this->used_attribute_point_channel_offset_[corresponding_attribute_idx];
				}
				CHECK_GE(cur_valid_point_channel_offset,0);

				if(cur_point_ignore || anno_type[scale_base_id] == caffe::ANNO_IGNORE){
					key_point_valid_len = MAX(key_point_valid_len,
							round(this->ignore_flag_radius_ * scale / Dtype(this->heat_map_a_)));
					cur_valid_point_channel_offset += this->channel_point_ignore_point_channel_offset_ ;
//					if(cur_point_ignore || anno_type[scale_base_id] == caffe::ANNO_IGNORE)
//					LOG(INFO)<<"Cur pixel anno_ignore , cur_point_ignore:"<<cur_point_ignore
//							<<"   and anno_type[scale_base_id] == caffe::ANNO_IGNORE: "<< (anno_type[scale_base_id] == caffe::ANNO_IGNORE);
				}else{
					key_point_valid_len = MAX(key_point_valid_len,
							round(this->key_point_valid_distance_ * scale / Dtype(this->heat_map_a_)));
				}
				int key_point_loc_diff_len = round(this->key_point_loc_diff_radius_ * scale / Dtype(this->heat_map_a_));
				int max_loop_len = max(key_point_valid_len,key_point_loc_diff_len);

				for(int dy = -max_loop_len; dy <= max_loop_len; ++dy){
					Dtype pt_y = heat_map_y + dy;
					if (round(pt_y) < 0 || round(pt_y) >=  prefetch_label.height())
						continue;
					for(int dx = -max_loop_len; dx <= max_loop_len; ++dx){
						Dtype pt_x = heat_map_x + dx;
						if ( round(pt_x) < 0 || round(pt_x) >=  prefetch_label.width())
							continue;
						/*
						 * Check which label need to be generated
						 */
						bool need_score = (dx*dx + dy*dy) <= (key_point_valid_len * key_point_valid_len);
						bool need_loc_diff = (this->need_point_loc_diff_) //&& (this->ignore_key_point_idxs_[point_idx] == false)
												&& ( (dx*dx + dy*dy) <= (key_point_loc_diff_len * key_point_loc_diff_len)) ;
						int offset = prefetch_label.offset(item_id,channel_point_base_offset_+cur_valid_point_channel_offset,round(pt_y), round(pt_x));
						if(need_score){
							top_label[offset] = 1;
//							LOG(INFO)<<"good  label 1 is generated at "<< pt_x <<","<<pt_y<<" at channel"<<cur_channel_offset;
							if(cur_has_attribute){
								Dtype attribute_value = coords_of_one_instance[corresponding_attribute_idx*2];
								int attribute_offset = prefetch_label.offset(item_id,channel_point_base_offset_+cur_attribute_channel_offset,
										round(pt_y), round(pt_x));
								top_label[attribute_offset] = attribute_value;
//								LOG(INFO)<<"has attribute value:"<< attribute_value;
							}
						}
						if(need_loc_diff){
							int diff_offset =prefetch_label.offset(item_id,channel_point_base_offset_+cur_diff_channel_offset,round(pt_y), round(pt_x));
							top_label[diff_offset            ] = Dtype(round(pt_x) - heat_map_x);
							top_label[diff_offset + map_size ] = Dtype(round(pt_y) - heat_map_y);
						}



					}
				}
			}
		}
	}

}

template <typename Dtype>
void IImageDataKeyPoint<Dtype>::LabelToVisualizedCVMat(const Blob<Dtype>& label, const int valid_point_id,
		cv::Mat& out_probs, cv::Mat& ignore_out_probs,int item_id,
		int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
		bool need_regression, Dtype heat_map_a  , Dtype heat_map_b   )
{
	color_channel_weight[0] = MIN(1, MAX(0, color_channel_weight[0] ));
	color_channel_weight[1] = MIN(1, MAX(0, color_channel_weight[1] ));
	color_channel_weight[2] = MIN(1, MAX(0, color_channel_weight[2] ));
	int map_size = label.offset(0,1,0,0);
	char value_c[100];
	const Dtype * top_label = label.cpu_data();
	int n_valid_point_size = used_key_point_idxs_.size();
	for (int  h = 0; h < label.height(); ++h)
	{
		for(int w = 0 ; w < label.width(); ++ w)
		{

			int top_idx = label.offset(item_id, this->channel_point_base_offset_, h , w);
			int label_base_offset  = top_idx + (channel_point_valid_keypoint_channel_offset_+ scale_base_id*n_valid_point_size +valid_point_id) * map_size;
			int ignore_label_base_offset =top_idx + (channel_point_ignore_point_channel_offset_ + scale_base_id*n_valid_point_size +valid_point_id )* map_size;
			int point_loc_diff_base_offset = top_idx + (channel_point_loc_diff_offset_ + (scale_base_id*n_valid_point_size +valid_point_id ) *2 )* map_size;
			int attribute_base_offset = top_idx + (channel_point_attribute_point_channel_offset_ + (scale_base_id*n_valid_point_size +valid_point_id )   )* map_size;

			const int tmp_h = h * heat_map_a + heat_map_b;
			const int tmp_w = w * heat_map_a + heat_map_b;
			CHECK(this->CheckValidIndexInRange(out_probs, tmp_h,tmp_w));
			CHECK(this->CheckValidIndexInRange(ignore_out_probs, tmp_h,tmp_w));
//			LOG(INFO)<<"show label channel "<<(channel_point_valid_keypoint_channel_offset_+ scale_base_id*n_valid_point_size +valid_point_id) ;
			if(top_label[label_base_offset ] != Dtype(0)  && top_label[label_base_offset ] > threshold)
			{
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = static_cast<uint8_t>(color_channel_weight[0] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(color_channel_weight[1] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));
				out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = static_cast<uint8_t>(color_channel_weight[2] * 255 * MIN(1, MAX(0, top_label[ label_base_offset ])));
//				LOG(INFO)<<"good    label is 1";

				if(this->need_point_loc_diff_ && need_regression && top_label[point_loc_diff_base_offset]  != Dtype(0)){
					Dtype coord[2];
					coord[0] = tmp_w- top_label[point_loc_diff_base_offset] * heat_map_a ;
					coord[0] = MIN(out_probs.cols, MAX(0, coord[0]));
					coord[1] = tmp_h- top_label[point_loc_diff_base_offset+ map_size*1]* heat_map_a ;
					coord[1] = MIN(out_probs.rows, MAX(0, coord[1]));

					cv::circle(out_probs, cv::Point( round(coord[0]), round(coord[1])), 1, cv::Scalar(255, 255, 0),1);
					out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 255;
				}
				if(this->used_attribute_point_idxs_.size() > 0 && top_label[attribute_base_offset]  != Dtype(0)){
					sprintf(value_c, "%.3f", top_label[attribute_base_offset]);
					cv::putText(out_probs, value_c, cv::Point( 0, tmp_h),
										CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0));
				}

			}

			if(top_label[ignore_label_base_offset] != Dtype(0)  && top_label[ignore_label_base_offset ] > threshold  )
			{
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(255 * MIN(1, MAX(0, top_label[ ignore_label_base_offset ])));
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[0] = 0;
				ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 0;
				if(this->need_point_loc_diff_ && need_regression && top_label[point_loc_diff_base_offset]  != Dtype(0)){
					Dtype coord[2];
					coord[0] = tmp_w- top_label[point_loc_diff_base_offset] * heat_map_a ;
					coord[0] = MIN(ignore_out_probs.cols, MAX(0, coord[0]));
					coord[1] = tmp_h- top_label[point_loc_diff_base_offset+ map_size*1]* heat_map_a ;
					coord[1] = MIN(ignore_out_probs.rows, MAX(0, coord[1]));

					cv::circle(ignore_out_probs, cv::Point( round(coord[0]), round(coord[1])), 1, cv::Scalar(255, 255, 0),1);
					ignore_out_probs.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 255;
				}
			}
		}
	}
}


/**
 * TODO
 */
template <typename Dtype>
void IImageDataKeyPoint<Dtype>:: PrintPic(int item_id, const string & output_path,
		cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr, const pair< string, vector<Dtype> > & cur_sample,
		const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label)
{
	cv::Mat & cv_img = *( (img_cv_ptr));
	string out_sample_name = IImageDataProcessor<Dtype>::GetPrintSampleName(cur_sample,sample_type);
	char path[512];
//	cv::Size output_size(this->out_width_, this->out_height_);
	cv::Size output_size(this->input_width_, this->input_height_);
//
//	LOG(INFO)<<"channel_point_base_offset_:"<<channel_point_base_offset_<<" "<<
//			"channel_point_valid_keypoint_channel_offset_:"<<channel_point_valid_keypoint_channel_offset_<<" "<<
//			"channel_point_loc_diff_offset_:"<<channel_point_loc_diff_offset_<<" "<<
//			"channel_point_ignore_point_channel_offset_:"<<channel_point_ignore_point_channel_offset_<<" "<<
//			"channel_point_channel_all_need_:"<<channel_point_channel_all_need_;
	Dtype color_channel_weight[3] = {0,1,0};

	for(int scale_base_id = 0 ; scale_base_id < this->scale_base_.size(); ++ scale_base_id)
	{
		for(int point_id = 0; point_id < this->used_key_point_idxs_.size(); ++point_id){

			cv::Mat out_probs, ignore_out_probs;
			cv::resize(cv_img, out_probs,output_size);
			cv::resize(cv_img, ignore_out_probs,output_size);

//			LabelToVisualizedCVMat(prefetch_label, point_id,out_probs, ignore_out_probs,item_id, scale_base_id,color_channel_weight, -1);
			LabelToVisualizedCVMat(prefetch_label, point_id,out_probs, ignore_out_probs,item_id, scale_base_id,color_channel_weight, -1,
					true, this->heat_map_a_, this->heat_map_b_);
			sprintf(path, "%s/pic/point_%03d_item_%03d_%s_s%f_s%f_point_label_scalebase.jpg", output_path.c_str(),point_id,item_id,
					out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			LOG(INFO)<< "Saving map: " << path;
			imwrite(path, out_probs);
			sprintf(path, "%s/pic/point_%03d_item_%03d_%s_s%f_s%f_point_ignore_scalebase.jpg", output_path.c_str(),point_id,item_id,
							out_sample_name.c_str(), float(scale),float(this->scale_base_[scale_base_id]));
			LOG(INFO)<< "Saving  map: " << path;
			imwrite(path, ignore_out_probs);
		}
	}
}

template <typename Dtype>
void IImageDataKeyPoint<Dtype>:: ShowDataAndPredictedLabel(const string & output_path,const string & img_name,
		const Blob<Dtype>& data, const int sampleID,const Dtype* mean_bgr,const Blob<Dtype>& label
		,const Blob<Dtype>& predicted,Dtype threshold){
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
		for(int point_id = 0; point_id < this->used_key_point_idxs_.size(); ++point_id){
			cv::Mat cv_img_original = caffe::BlobImgDataToCVMat(data, sampleID, mean_bgr[0],
					mean_bgr[1],mean_bgr[2]);
			cv::Mat original_clone, original_clone_ignore ;
			cv::Mat cv_img_original_mix;
			cv::resize(cv_img_original, cv_img_original_mix, label_size);
			cv::Mat cv_img_original_gt_pred(cv_img_original.rows, cv_img_original.cols * 2 + this->PIC_MARGIN, CV_8UC3);


			cv::Mat cv_img_gt = cv::Mat(label.height(), label.width(), CV_8UC3);
			cv_img_gt = cv::Mat::zeros(label.height(), label.width(), CV_8UC3);
			cv::Mat cv_img_gt_ignore = cv_img_gt.clone();
			cv::Mat cv_img_pred = cv_img_gt.clone();

			color_channel_weight[0] = 1;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 1;
			LabelToVisualizedCVMat(label,point_id, cv_img_gt, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, -1,false );
			original_clone = cv_img_original.clone();
			original_clone_ignore = cv_img_original.clone();
			/**
			 * the ground truth information is stored in the second channel
			 */
			color_channel_weight[0] = 0;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 0;
			LabelToVisualizedCVMat(label, point_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, -1,
					true,this->heat_map_a_, this->heat_map_b_ );
			LabelToVisualizedCVMat(label, point_id,cv_img_original_mix, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, -1,false );
			original_clone.copyTo(cv_img_original_gt_pred(
					cv::Rect(0,0,original_clone.cols, original_clone.rows)));

			color_channel_weight[0] = 1;
			color_channel_weight[1] = 1;
			color_channel_weight[2] = 1;
			LabelToVisualizedCVMat(predicted,point_id, cv_img_pred, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, threshold,false );

			original_clone = cv_img_original.clone();
			/**
			 * the prediction information is stored in the first channel
			 */
			color_channel_weight[0] = 1;
			color_channel_weight[1] = 0;
			color_channel_weight[2] = 0;
			LabelToVisualizedCVMat(predicted, point_id,original_clone, original_clone_ignore,sampleID, scale_base_id,color_channel_weight, threshold,
					true,this->heat_map_a_, this->heat_map_b_ );
			LabelToVisualizedCVMat(predicted, point_id,cv_img_original_mix, cv_img_gt_ignore,sampleID, scale_base_id,color_channel_weight, threshold,false );
			original_clone.copyTo(cv_img_original_gt_pred(cv::Rect(
					original_clone.cols + this->PIC_MARGIN, 0,
					original_clone.cols, original_clone.rows)));

			sprintf(path, "%s/%s_bbox_original_gt_pred_point_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str(),point_id,scale_base_id );
			imwrite(path, cv_img_original_gt_pred, params_jpg);
			sprintf(path, "%s/%s_bbox_gt_pred_point_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,point_id,scale_base_id);
			imwrite(path, cv_img_original_mix, params_jpg);
			sprintf(path, "%s/%s_bbox_gt_point_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,point_id,scale_base_id);
			imwrite(path, cv_img_gt, params_jpg);
			sprintf(path, "%s/%s_bbox_pred_point_%03d_base_%d.jpg", output_path.c_str(),img_name.c_str() ,point_id,scale_base_id);
			imwrite(path, cv_img_pred, params_jpg);
		}
	}

}


INSTANTIATE_CLASS(IImageDataKeyPoint);
}  // namespace caffe

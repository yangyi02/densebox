#include <string>
#include <vector>
#include <algorithm>
#include "caffe/layers/pyramid_data_layers.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"

namespace caffe {

template <typename Dtype>
ROIDataLayer<Dtype>::ROIDataLayer(const LayerParameter& param): Layer<Dtype>(param){
	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
}

template <typename Dtype>
void ROIDataLayer<Dtype>::ShuffleIds(vector<int>& ids){
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());
	shuffle(ids.begin(), ids.end(), prefetch_rng);
}

template <typename Dtype>
void ROIDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	const ROIDataParam& roi_data_param = this->layer_param_.roi_data_param();
	pos_iou_ratio_ = roi_data_param.pos_iou_ratio();
	CHECK_GE(pos_iou_ratio_,0);
	CHECK_LE(pos_iou_ratio_,1);
	neg_iou_ratio_ = roi_data_param.neg_iou_ratio();
	CHECK_GE(neg_iou_ratio_,0);
	CHECK_LE(neg_iou_ratio_,1);
	num_class_ = roi_data_param.num_class();
	CHECK_GE(num_class_,1);
	need_balance_ = roi_data_param.need_balance();
	neg_ratio_ = roi_data_param.neg_ratio();
	hard_ratio_ = roi_data_param.hard_ratio();
	hard_threshold_ = roi_data_param.hard_threshold();

	CHECK_GE(hard_threshold_,0);
	CHECK_LE(hard_threshold_,neg_iou_ratio_);
	neg_ratio_ = std::min(  std::max(neg_ratio_,Dtype(0)),Dtype(1));
	hard_ratio_ = std::min( std::max(hard_ratio_,Dtype(0)),Dtype(1));
	CHECK_EQ(bottom.size()+1, top.size());
	has_predicted_label_ = (bottom.size()>3);
	predict_dim_ = 0;
	if(has_predicted_label_){
		predict_dim_ = bottom[3]->width();
	}
}


template <typename Dtype>
void ROIDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->num(),bottom[1]->num());

	ROI_data_length_ = bottom[0]->width();
	ROI_info_length_ = bottom[1]->width();
	GT_data_length_  = bottom[2]->width();
	top[0]->Reshape(1,1,1,ROI_data_length_);
	top[1]->Reshape(1,1,1,ROI_info_length_);
	top[2]->Reshape(1,1,1,1);
	top[3]->Reshape(1,1,1,4*num_class_);
	if(has_predicted_label_){
		CHECK_EQ(bottom[0]->num(),bottom[3]->num());
		top[4]->Reshape(1,1,1,1);
	}
}

/**
 * @brief Generate ROI labels and target_bbox_diff from ground truth BBox and given ROIs and ROI_info.
 *		  The output has 4 blobs, top[0] is the filtered ROI, and top[1] is the ROI_info.
 *		  top[2] and top[3] output the ROI labels and target_bbox_diff.
 *		  ROI label is a non-negative integer, with 0 indicating background.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 *
 *  @TODO Balance positive and negative ?
 */
template <typename Dtype>
void ROIDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	FilterInputROIsWithIOU(bottom);
	bool balance_successful = false;
	if(need_balance_){
		balance_successful = BalanceROIs(bottom);
	}

	if(balance_successful == false){
		CHECK_EQ(valid_ROIs_.size()%ROI_data_length_,0);
		top[0]->Reshape(valid_ROIs_.size()/ROI_data_length_,1,1,ROI_data_length_);
		caffe::caffe_copy(valid_ROIs_.size(),&(valid_ROIs_[0]),top[0]->mutable_cpu_data());

		CHECK_EQ(valid_ROI_info_.size()%ROI_info_length_,0);
		top[1]->Reshape(valid_ROI_info_.size()/ROI_info_length_,1,1,ROI_info_length_);
		caffe::caffe_copy(valid_ROI_info_.size(),&(valid_ROI_info_[0]),top[1]->mutable_cpu_data());

		top[2]->Reshape(valid_ROIs_label_.size() ,1,1,1);
		caffe::caffe_copy(valid_ROIs_label_.size(),&(valid_ROIs_label_[0]),top[2]->mutable_cpu_data());
		int out_n_rois = valid_ROIs_label_.size();

		CHECK_EQ(valid_ROIs_bbox_diff_.size()%4,0);
		top[3]->Reshape(valid_ROIs_bbox_diff_.size()/4,1,1,4 * num_class_);
		caffe::caffe_set(top[3]->count(),Dtype(0),top[3]->mutable_cpu_data());

		Dtype* out_ROI_bbox_diff = &(valid_ROIs_bbox_diff_[0]);
		Dtype* out_ROI_label = top[2]->mutable_cpu_data();
		Dtype* top_3_data = top[3]->mutable_cpu_data();
		for(int i=0; i < out_n_rois; ++i){
			int class_id = out_ROI_label[i] - 1;
			if(class_id >= 0)
				caffe::caffe_copy(4,out_ROI_bbox_diff + i * 4, top_3_data + i*4*num_class_ + class_id);
		}

		if(has_predicted_label_){
			top[4]->Reshape(bottom[3]->shape());
			caffe::caffe_copy(bottom[3]->count(),bottom[3]->cpu_data(),top[4]->mutable_cpu_data());
		}
	}else{
		CHECK_EQ(out_ROIs_.size()%ROI_data_length_,0);
		top[0]->Reshape(out_ROIs_.size()/ROI_data_length_,1,1,ROI_data_length_);
		caffe::caffe_copy(out_ROIs_.size(),&(out_ROIs_[0]),top[0]->mutable_cpu_data());

		CHECK_EQ(out_ROI_info_.size()%ROI_info_length_,0);
		top[1]->Reshape(out_ROI_info_.size()/ROI_info_length_,1,1,ROI_info_length_);
		caffe::caffe_copy(out_ROI_info_.size(),&(out_ROI_info_[0]),top[1]->mutable_cpu_data());

		top[2]->Reshape(out_ROIs_label_.size() ,1,1,1);
		caffe::caffe_copy(out_ROIs_label_.size(),&(out_ROIs_label_[0]),top[2]->mutable_cpu_data());
		int out_n_rois = out_ROIs_label_.size();

		CHECK_EQ(out_ROIs_bbox_diff_.size()%4,0);
		top[3]->Reshape(out_ROIs_bbox_diff_.size()/4,1,1,4 * num_class_);
		caffe::caffe_set(top[3]->count(),Dtype(0),top[3]->mutable_cpu_data());

		Dtype* out_ROI_bbox_diff = &(out_ROIs_bbox_diff_[0]);
		Dtype* out_ROI_label = top[2]->mutable_cpu_data();
		Dtype* top_3_data = top[3]->mutable_cpu_data();
		for(int i=0; i < out_n_rois; ++i){
			int class_id = out_ROI_label[i] - 1;
			if(class_id >= 0)
				caffe::caffe_copy(4,out_ROI_bbox_diff + i * 4, top_3_data + i*4*num_class_ + class_id);
		}
		if(has_predicted_label_){
			CHECK_EQ(out_ROIs_predict_.size()%bottom[3]->width(),0);
			top[4]->Reshape(out_ROIs_predict_.size()/bottom[3]->width(),1,1,bottom[3]->width());
			caffe::caffe_copy(out_ROIs_predict_.size(),&(out_ROIs_predict_[0]),top[4]->mutable_cpu_data());
			CHECK_EQ(top[4]->num(),top[0]->num());
		}
	}

}

template <typename Dtype>
unsigned int ROIDataLayer<Dtype>::PrefetchRand()
{
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	return (*prefetch_rng)();
}

template <typename Dtype>
Dtype ROIDataLayer<Dtype>::PrefetchRandFloat()
{
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	return  (*prefetch_rng)() / static_cast<Dtype>(prefetch_rng->max());
}
/**
 * balance ROIs based on the positive ratio.  Return false if no need to balance.
 */
template <typename Dtype>
bool ROIDataLayer<Dtype>::BalanceROIs(const vector<Blob<Dtype>*>& bottom){

	const int gt_num = bottom[2]->num();

	const Dtype jitter_ratio = 1-sqrt(pos_iou_ratio_);
	const Dtype* in_GT_data = bottom[2]->cpu_data();
	/**
	 * add bbox generated from GT ， and one neg
	 */
	valid_IOUs_.push_back(0);
	valid_ROIs_label_.push_back(0) ;
	valid_ROIs_bbox_diff_.push_back(0);
	valid_ROIs_bbox_diff_.push_back(0);
	valid_ROIs_bbox_diff_.push_back(0);
	valid_ROIs_bbox_diff_.push_back(0);

	valid_ROIs_.push_back(0);
	valid_ROIs_.push_back(0);
	valid_ROIs_.push_back(0);
	valid_ROIs_.push_back(0);
	valid_ROIs_.push_back(0);

	valid_ROI_info_.push_back(0);
	valid_ROI_info_.push_back(0);
	valid_ROI_info_.push_back(0);
	valid_ROI_info_.push_back(0);

	for(int i=0; i < gt_num; ++i){
		int cur_GT_data_start = i* GT_data_length_;
		Dtype class_id = in_GT_data[cur_GT_data_start];
		Dtype item_id = in_GT_data[cur_GT_data_start+1];
		Dtype x1 = in_GT_data[cur_GT_data_start+2];
		Dtype y1 = in_GT_data[cur_GT_data_start+3];
		Dtype x2 = in_GT_data[cur_GT_data_start+4];
		Dtype y2 = in_GT_data[cur_GT_data_start+5];
		Dtype width = x2-x1;
		Dtype height = y2 - y1;
		Dtype x_jitter = width*this->PrefetchRandFloat()*jitter_ratio*(this->PrefetchRandFloat()>0.5)  ;
		Dtype y_jitter = height*this->PrefetchRandFloat()*jitter_ratio*(this->PrefetchRandFloat()>0.5) ;
		// @TODO   a probability to change proposal class ?
		Dtype overlap = caffe::GetOverlap(x1+x_jitter,y1+y_jitter,x2+x_jitter,y2+y_jitter,
					x1,y1,x2,y2,caffe::OVERLAP_UNION);
		valid_IOUs_.push_back(overlap);
		valid_ROIs_label_.push_back(class_id + 1) ;
		valid_ROIs_bbox_diff_.push_back(x_jitter);
		valid_ROIs_bbox_diff_.push_back(y_jitter);
		valid_ROIs_bbox_diff_.push_back(x_jitter);
		valid_ROIs_bbox_diff_.push_back(y_jitter);

		valid_ROIs_.push_back(item_id);
		valid_ROIs_.push_back(x1+x_jitter);
		valid_ROIs_.push_back(y1+y_jitter);
		valid_ROIs_.push_back(x2+x_jitter);
		valid_ROIs_.push_back(y2+y_jitter);

		valid_ROI_info_.push_back(class_id);
		valid_ROI_info_.push_back(int((x1+x2)/2));
		valid_ROI_info_.push_back(int((y1+y2)/2));
		valid_ROI_info_.push_back(1);

	}



	out_ROIs_.clear();
	out_ROI_info_.clear();
	out_ROIs_label_.clear();
	out_ROIs_bbox_diff_.clear();
	out_ROIs_predict_.clear();

	pos_ids_.clear();
	normal_neg_ids_.clear();
	hard_neg_ids_.clear();
	selected_ids_.clear();
	for(int i=0; i < valid_IOUs_.size(); ++i){
		if(valid_IOUs_[i] <= hard_threshold_){
			normal_neg_ids_.push_back(i);
		}else if(valid_IOUs_[i] >= pos_iou_ratio_){
			pos_ids_.push_back(i);
		}else {
			hard_neg_ids_.push_back(i);
		}
	}

	bool is_pos_fewer = true;
	int pos_needed_num = pos_ids_.size();

	int total_neg_needed_num = MIN( (pos_ids_.size()+0.0)/(1-neg_ratio_),
			normal_neg_ids_.size()+hard_neg_ids_.size());
	total_neg_needed_num = MAX(0,total_neg_needed_num);
	if(total_neg_needed_num == normal_neg_ids_.size()+hard_neg_ids_.size()){
		is_pos_fewer  = false;
		pos_needed_num  = (normal_neg_ids_.size()+hard_neg_ids_.size()+0.0)/neg_ratio_*(1-neg_ratio_);
		pos_needed_num = MAX(0,pos_needed_num);
		pos_needed_num = MIN(pos_needed_num,pos_ids_.size());
		total_neg_needed_num = MIN( (pos_needed_num+0.0)/(1-neg_ratio_),
					normal_neg_ids_.size()+hard_neg_ids_.size());
		total_neg_needed_num = MAX(0,total_neg_needed_num);
	}

	int hard_needed_num =  total_neg_needed_num * hard_ratio_;
	int normal_neg_needed = total_neg_needed_num - hard_needed_num;

	// if hard_neg is not enough, sample more from normal_neg
	int actual_hard_sampled = MIN(hard_needed_num, hard_neg_ids_.size());
	normal_neg_needed += MAX(hard_needed_num - actual_hard_sampled,0);

	// if normal_neg is not enough, sample more from hard_neg
	int actual_normal_neg_sampled =  MIN(normal_neg_needed, normal_neg_ids_.size());
	hard_needed_num += MAX(normal_neg_needed - actual_normal_neg_sampled,0);
	actual_hard_sampled = MIN(hard_needed_num, hard_neg_ids_.size());


	//  copy the sampled rois
	int total_sample_num = actual_hard_sampled + actual_normal_neg_sampled + pos_needed_num;
	if(total_sample_num == 0){
		pos_needed_num++;
		actual_hard_sampled+= hard_neg_ids_.size()>0;
		total_sample_num = actual_hard_sampled + actual_normal_neg_sampled + pos_needed_num;
	}

	out_ROIs_.resize(total_sample_num * ROI_data_length_);
	out_ROI_info_.resize(total_sample_num * ROI_info_length_);
	out_ROIs_label_.resize(total_sample_num);
	out_ROIs_bbox_diff_.resize(total_sample_num*4);

	LOG(INFO)<<"In ROI_data_balance, total_sample_num = "<<total_sample_num<<
			"  actual_hard_sampled = "<<actual_hard_sampled<<"  actual_normal_neg_sampled = "<<
			actual_normal_neg_sampled<<"  pos_needed_num = "<<pos_needed_num;


	ShuffleIds(pos_ids_);
	ShuffleIds(normal_neg_ids_);
	ShuffleIds(hard_neg_ids_);

	int* ids_ptr = &(pos_ids_[0]);

	for(int i=0; i < pos_needed_num; ++i){
		selected_ids_.push_back(*(ids_ptr++));
	}

	ids_ptr = &(normal_neg_ids_[0]);
	for(int i=0; i < actual_normal_neg_sampled; ++i){
		selected_ids_.push_back(*(ids_ptr++));
	}

	ids_ptr = &(hard_neg_ids_[0]);
	for(int i=0; i < actual_hard_sampled; ++i){
		selected_ids_.push_back(*(ids_ptr++));
	}

	// copy balanced data to output

	Dtype * bottom_predicted = NULL;
	Dtype * out_predicted = NULL;
	if(has_predicted_label_){
		bottom_predicted = bottom[3]->mutable_cpu_data();
		out_ROIs_predict_.resize(total_sample_num * predict_dim_);
		out_predicted = &(out_ROIs_predict_[0]);
	}
	for(int i=0; i < selected_ids_.size(); ++i){
		int id = selected_ids_[i];
		copy(valid_ROIs_.begin()+id*ROI_data_length_, valid_ROIs_.begin()+(id+1)*ROI_data_length_,
				out_ROIs_.begin() + i*ROI_data_length_);
		copy(valid_ROI_info_.begin()+id*ROI_info_length_, valid_ROI_info_.begin()+(id+1)*ROI_info_length_,
				out_ROI_info_.begin() + i*ROI_info_length_);
		out_ROIs_label_[i] = valid_ROIs_label_[id];
		copy(valid_ROIs_bbox_diff_.begin()+id*4, valid_ROIs_bbox_diff_.begin()+(id+1)*4,
				out_ROIs_bbox_diff_.begin() + i*4);
		if(has_predicted_label_){
			caffe::caffe_copy(predict_dim_,bottom_predicted + id*predict_dim_,
					out_predicted + i*predict_dim_);
		}
	}
	return true;
}


/**
 * @TODO   a probability to change proposal class ?
 */
template <typename Dtype>
void ROIDataLayer<Dtype>::FilterInputROIsWithIOU(const vector<Blob<Dtype>*>& bottom){
	valid_ROIs_.clear();
	valid_ROI_info_.clear();
	valid_ROIs_label_.clear();
	valid_ROIs_bbox_diff_.clear();
	valid_IOUs_.clear();
	const int input_num = bottom[0]->num();
	const int gt_num = bottom[2]->num();

	int pos_num = 0;
	int neg_num = 0;
	int ignore_num = 0;

	if(input_num <= 0)
		return;
	Dtype* in_ROIs_data = bottom[0]->mutable_cpu_data();
	Dtype* in_ROIs_info_data = bottom[1]->mutable_cpu_data();
	// if grund_truth has no bbox, it means all ROIs should be negative.
	if(gt_num <= 0){
		for(int i=0; i < input_num; ++i){
			for(int j=0; j < ROI_data_length_; ++j){
				valid_ROIs_.push_back(*(in_ROIs_data++));
			}
			for(int j=0; j < ROI_info_length_; ++j){
				valid_ROI_info_.push_back(*(in_ROIs_info_data++));
			}
			valid_IOUs_.push_back(0);
			valid_ROIs_label_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
		}
		return ;
	}
	const Dtype* in_GT_data = bottom[2]->cpu_data();

	//Get ROIs according to IOU
	for(int i=0; i < input_num; ++i){
		Dtype* cur_ROI_data = in_ROIs_data + i * ROI_data_length_;
		Dtype* cur_ROI_info_data = in_ROIs_info_data + i * ROI_info_length_;
		int max_overlap_idx = -1;
		Dtype max_overlap = -1;
		for(int j = 0; j < gt_num; ++j ){
			const int cur_GT_data_start = j * GT_data_length_;

			// @TODO   a probability to change proposal class ?
			Dtype overlap = caffe::GetOverlap(cur_ROI_data[1],cur_ROI_data[2],cur_ROI_data[3],cur_ROI_data[4],
					in_GT_data[cur_GT_data_start+2],in_GT_data[cur_GT_data_start+3],
					in_GT_data[cur_GT_data_start+4],in_GT_data[cur_GT_data_start+5],caffe::OVERLAP_UNION);
			bool is_class_matched = int(in_GT_data[cur_GT_data_start ]) == int(cur_ROI_info_data[0]);
			if(overlap > max_overlap && is_class_matched){
				max_overlap = overlap;
				max_overlap_idx = j;
			}

		}

		if(max_overlap <= neg_iou_ratio_){
			for(int j=0; j < ROI_data_length_; ++j){
				valid_ROIs_.push_back( cur_ROI_data[j]);
			}
			for(int j=0; j < ROI_info_length_; ++j){
				valid_ROI_info_.push_back( cur_ROI_info_data[j]);
			}
			valid_IOUs_.push_back(MAX(max_overlap,0));
			valid_ROIs_label_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			valid_ROIs_bbox_diff_.push_back(0);
			neg_num++;
		} else if(max_overlap > pos_iou_ratio_){
			CHECK_GE(max_overlap_idx,0);
			CHECK_GE(max_overlap,0);
			//			LOG(INFO)<<"overlap: "<<overlap;
			const int cur_GT_data_start = max_overlap_idx * GT_data_length_;
			for(int j=0; j < ROI_data_length_; ++j){
				valid_ROIs_.push_back( cur_ROI_data[j]);
			}
			for(int j=0; j < ROI_info_length_; ++j){
				valid_ROI_info_.push_back( cur_ROI_info_data[j]);
			}

//			{
//				Dtype* cur_ROI_info_data = in_ROIs_info_data + i * ROI_info_length_;
//				LOG(INFO)<<"pos in roi_data_layer at loc: ("<< cur_ROI_info_data[2]<<","<<
//						cur_ROI_info_data[1]<<").";
//			}
			valid_IOUs_.push_back(MAX(max_overlap,0));
			valid_ROIs_label_.push_back(in_GT_data[cur_GT_data_start] + 1) ;

			valid_ROIs_bbox_diff_.push_back(cur_ROI_data[1] - in_GT_data[cur_GT_data_start+2]);
			valid_ROIs_bbox_diff_.push_back(cur_ROI_data[2] - in_GT_data[cur_GT_data_start+3]);
			valid_ROIs_bbox_diff_.push_back(cur_ROI_data[3] - in_GT_data[cur_GT_data_start+4]);
			valid_ROIs_bbox_diff_.push_back(cur_ROI_data[4] - in_GT_data[cur_GT_data_start+5]);
			pos_num ++;
//			{
//
//				LOG(INFO)<<"loc_diff : ("<< cur_ROI_data[1] - in_GT_data[cur_GT_data_start+2] <<" "<<
//						cur_ROI_data[2] - in_GT_data[cur_GT_data_start+3]<<" "<<
//						cur_ROI_data[3] - in_GT_data[cur_GT_data_start+4]<< "  "<<
//						cur_ROI_data[4] - in_GT_data[cur_GT_data_start+5]<<") , with score "<< cur_ROI_info_data[3];
//			}
		}
		else{
			ignore_num++;
		}
//		LOG(INFO)<<"find roi "<<valid_ROIs_label_.size()-1 <<" with label "<<valid_ROIs_label_[valid_ROIs_label_.size()-1];
	}
	LOG(INFO)<<"pos_num: "<<pos_num <<"  neg_num: "<<neg_num<<" ignore_num:"<<ignore_num;
}

template <typename Dtype>
void ROIDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(ROIDataLayer);
#endif



INSTANTIATE_CLASS(ROIDataLayer);
REGISTER_LAYER_CLASS(ROIData);


}  // namespace caffe

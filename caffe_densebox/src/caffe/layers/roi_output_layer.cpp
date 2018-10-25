#include <string>
#include <vector>
#include <algorithm>
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
void ROIOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	CHECK(this->layer_param_.has_detection_output_param());
	const DetectionOutputParameter& detection_output_param = this->layer_param_.detection_output_param();
	threshold_ = detection_output_param.threshold();
	nms_need_nms_ = detection_output_param.nms_param().need_nms();
	nms_overlap_ratio_ = detection_output_param.nms_param().overlap_ratio();
	nms_top_n_ = detection_output_param.nms_param().top_n();
	nms_add_score_ = detection_output_param.nms_param().add_score();
	channel_per_scale_ = detection_output_param.channel_per_scale();
	CHECK_GE(channel_per_scale_,5)<<"channel_per_scale_ should be at least 5: (score,dx1,dy1,dx2,dy2)";
	num_class_ = detection_output_param.num_class();
	CHECK_GT(num_class_, 0);
	class_names_.clear();
	if(num_class_ > 1){
		CHECK(detection_output_param.has_class_name_list());
		string file_list_name = detection_output_param.class_name_list();
		LOG(INFO) << "Opening class list file " << file_list_name;
		std::ifstream infile(file_list_name.c_str());
		CHECK(infile.good());
		string class_name;
		while(std::getline(infile,class_name)){
			if(class_name.empty()) continue;
			class_names_.push_back(class_name);
		}
		infile.close();

	}else{
		class_names_.push_back("main");
	}
	CHECK_EQ(class_names_.size(),num_class_);
	//	top[0]->Reshape(1,1,nms_top_n_,channel_per_scale_);
	all_candidate_bboxes_.clear();
	is_candidate_bbox_selected_.clear();
	output_bboxes_.clear();
	for(int class_id = 0; class_id < num_class_; ++class_id){
		all_candidate_bboxes_.push_back(vector<BBox<Dtype> >());
		output_bboxes_.push_back(vector<BBox<Dtype> >());
	}
	time_get_bbox_ = time_total_ = time_nms_ = 0;
	show_time_ = ((getenv("SHOW_TIME") != NULL) && (getenv("SHOW_TIME")[0] == '1'));

	bbox_data_size = 5;
	bbox_info_size = 4;
	is_bottom_rois_ = false;
	if(top.size() > 0){
		CHECK_EQ(top.size(),2);
		top[0]->Reshape(1, 1,1,bbox_data_size);
		top[1]->Reshape(1, 1,1,bbox_info_size);
		is_bottom_rois_ = (bottom.size() > 1);
	}

}


template <typename Dtype>
void ROIOutputLayer<Dtype>::GetROIsFromFeatureMap(const vector<Blob<Dtype>*>& bottom ){
	int num = bottom[0]->num();

	const int map_height = bottom[0]->height();
	const int map_width = bottom[0]->width();
	const int map_size = map_height* map_width;
	const int scale_num = bottom[0]->channels()/ channel_per_scale_;
	CHECK_EQ(bottom[0]->channels()% channel_per_scale_,0);
	// Check if num_class_ * 5 *  channel_per_scale_ * scale_num = bottom[0]->channels()
	CHECK_EQ((bottom[0]->channels()/scale_num)/5, this->num_class_ );
	CHECK_EQ((bottom[0]->channels()/scale_num)%5, 0 );
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	/**
	 * Get all bbox, and do nms if necessary
	 */
	for(int num_id=0; num_id < num; ++num_id){
		timer_get_bbox_.Start();
		for(int scale_id = 0; scale_id < scale_num; ++scale_id){
			for(int class_id = 0; class_id <  num_class_; ++class_id){
				// Get bbox candidates
				int score_channel = scale_id*channel_per_scale_ + class_id;
				int offset_channel = scale_id*channel_per_scale_ + num_class_ + class_id*4;
				Dtype* scores = bottom_data + bottom[0]->offset(num_id,score_channel  , 0,0);
				Dtype* dx1 =    bottom_data + bottom[0]->offset(num_id,offset_channel+0,0,0);
				Dtype* dy1 =    bottom_data + bottom[0]->offset(num_id,offset_channel+1,0,0);
				Dtype* dx2 =    bottom_data + bottom[0]->offset(num_id,offset_channel+2,0,0);
				Dtype* dy2 =    bottom_data + bottom[0]->offset(num_id,offset_channel+3,0,0);

				for(int off = 0; off< map_size; ++off){
					if(scores[off] < this->threshold_)
						continue;
					int h = off / map_width;
					int w = off % map_width ;
					BBox<Dtype> bbox;
					bbox.score = scores[off];
					bbox.x1 = w - dx1[off];
					bbox.y1 = h - dy1[off];
					bbox.x2 = w - dx2[off];
					bbox.y2 = h - dy2[off];
					bbox.center_h = h;
					bbox.center_w = w;
					bbox.id = num_id;
					all_candidate_bboxes_[class_id].push_back(bbox);
//					LOG(INFO)<<"("<<h<<","<<w<<")  has score "<< scores[off];
				}

			}
		}
		time_get_bbox_ += timer_get_bbox_.MilliSeconds();
		timer_nms_.Start();
		// do nms
		for(int class_id = 0; class_id < this->num_class_; ++class_id){
			vector<BBox<Dtype> > & cur_box_list = this->all_candidate_bboxes_[class_id];
			vector<BBox<Dtype> > & cur_outbox_list = this->output_bboxes_[class_id];

			this->is_candidate_bbox_selected_ = caffe::nms(cur_box_list,this->nms_overlap_ratio_,this->nms_top_n_,false);
			for(int i=0 ; i < this->is_candidate_bbox_selected_.size(); ++i){
				if(this->is_candidate_bbox_selected_[i]){
					cur_outbox_list.push_back(cur_box_list[i]);
				}
			}
			cur_box_list.clear();

		}
		time_nms_ += timer_nms_.MilliSeconds();
	}
}

template <typename Dtype>
void ROIOutputLayer<Dtype>::GetROIsFromROIs(const vector<Blob<Dtype>*>& bottom ){
	int num = bottom[0]->num();
	CHECK_EQ(num, bottom[1]->num());
	CHECK_EQ(bbox_data_size, bottom[0]->width());
	CHECK_EQ(bbox_info_size, bottom[1]->width());

	Dtype* ROI_data = bottom[0]->mutable_cpu_data();
	Dtype* ROI_info = bottom[1]->mutable_cpu_data();

	// initialize ROI_num_id_list_
	for(int i=0; i < ROI_num_id_list_.size(); ++i){
		ROI_num_id_list_[i].clear();
	}
	// Generate ROI_num_id_list_ from all ROIs
	for(int i=0; i < num; ++i){
		int item_id = ROI_data[i * bbox_data_size];
		while(item_id >= ROI_num_id_list_.size()){
			ROI_num_id_list_.push_back(vector<int>());
		}
		ROI_num_id_list_[item_id].push_back(i);
	}
	/**
	 * Get all bbox, and do nms if necessary
	 */
	for(int item_id=0; item_id < ROI_num_id_list_.size(); ++item_id){
		timer_get_bbox_.Start();
		vector<int>& cur_ROI_list = ROI_num_id_list_[item_id];
		int list_size  = cur_ROI_list.size();
		for(int i=0; i < list_size; ++i){
			int ROI_id = cur_ROI_list[i];
			if(ROI_info[ROI_id * bbox_info_size + 3] < threshold_ )
				continue;
			BBox<Dtype> bbox;
			Dtype* cur_ROI_data = ROI_data + ROI_id * bbox_data_size ;
			Dtype* cur_ROI_info = ROI_info + ROI_id * bbox_info_size ;

			bbox.id = *(cur_ROI_data++);
			bbox.x1 = *(cur_ROI_data++);
			bbox.y1 = *(cur_ROI_data++);
			bbox.x2 = *(cur_ROI_data++);
			bbox.y2 = *(cur_ROI_data++);
			CHECK_EQ(bbox.id,item_id);
			int cur_class_id =  *(cur_ROI_info++);
			bbox.center_w =     *(cur_ROI_info++);
			bbox.center_h =     *(cur_ROI_info++);
			bbox.score =        *(cur_ROI_info++);
			all_candidate_bboxes_[cur_class_id].push_back(bbox);
		}

		time_get_bbox_ += timer_get_bbox_.MilliSeconds();
		timer_nms_.Start();
		// do nms
		for(int class_id = 0; class_id < this->num_class_; ++class_id){
			vector<BBox<Dtype> > & cur_box_list = this->all_candidate_bboxes_[class_id];
			vector<BBox<Dtype> > & cur_outbox_list = this->output_bboxes_[class_id];

			this->is_candidate_bbox_selected_ = caffe::nms(cur_box_list,this->nms_overlap_ratio_,this->nms_top_n_,false);
			for(int i=0 ; i < this->is_candidate_bbox_selected_.size(); ++i){
				if(this->is_candidate_bbox_selected_[i]){
					cur_outbox_list.push_back(cur_box_list[i]);
				}
			}
			cur_box_list.clear();
		}
		time_nms_ += timer_nms_.MilliSeconds();
	}
}


template <typename Dtype>
void ROIOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	timer_total_.Start();
	time_bbox_to_blob_ = time_get_bbox_ = time_total_ = time_nms_ = 0;
	// initialize output_bboxes_ , all_candidate_bboxes_  output_bboxes_num_ids_
	for(int class_id = 0; class_id <  num_class_; ++class_id){
		output_bboxes_[class_id].clear();
		all_candidate_bboxes_[class_id].clear();
	}
	if(is_bottom_rois_){
		GetROIsFromROIs(bottom);
	}else {
		GetROIsFromFeatureMap(bottom);
	}

	/* BBoxes to Blob
	 * Each bbox is represented as:
	 * ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
	 *
	 * Each bbox_info is represented as (class_id,   w, h , score);
	 *
	 */
	int n_bbox = 0;
	for(int i=0; i < output_bboxes_.size(); ++i){
		n_bbox += output_bboxes_[i].size();
	}
	top[0]->Reshape(n_bbox, 1,1,bbox_data_size);
	top[1]->Reshape(n_bbox, 1,1,bbox_info_size);
	if(n_bbox <= 0){
		top[0]->Reshape(1, 1,1,bbox_data_size);
		top[1]->Reshape(1, 1,1,bbox_info_size);

		caffe::caffe_set(top[0]->count(),Dtype(0),top[0]->mutable_cpu_data());
		caffe::caffe_set(top[1]->count(),Dtype(0),top[1]->mutable_cpu_data());

		return;
	}
	timer_bbox_to_blob_.Start();
	Dtype* bbox_data = top[0]->mutable_cpu_data();
	Dtype* bbox_info = top[1]->mutable_cpu_data();
	for(int class_id = 0; class_id < this->num_class_; ++class_id){
		int cur_class_bbox_num = output_bboxes_[class_id].size();
		for(int i = 0; i < cur_class_bbox_num; ++i){
			BBox<Dtype>& cur_box = this->output_bboxes_[class_id][i];
			*(bbox_info++) = class_id;
			*(bbox_data++) = cur_box.id;
			*(bbox_info++) = cur_box.center_w;
			*(bbox_info++) = cur_box.center_h;
			*(bbox_data++) = cur_box.x1;
			*(bbox_data++) = cur_box.y1;
			*(bbox_data++) = cur_box.x2;
			*(bbox_data++) = cur_box.y2;
			*(bbox_info++) = cur_box.score;
//
//			LOG(INFO)<<"On top , ("<<cur_box.center_h<<","<<cur_box.center_w<<
//					")  has score "<< cur_box.score;
		}
	}
	time_bbox_to_blob_ += timer_bbox_to_blob_.MilliSeconds();
	time_total_ += timer_total_.MilliSeconds();
	if(this->show_time_){
		LOG(INFO)<<"Time cost for ROIOutputLayer forward : "<<time_total_<<" ms, in which get bbox cost "<<
				this->time_get_bbox_<<" ms, nms cost "<<this->time_nms_ <<"ms, bbox_to_blob cost"<<
				time_bbox_to_blob_<<" ms.";
	}
}


template <typename Dtype>
vector< BBox<Dtype> >& ROIOutputLayer<Dtype>::GetFilteredBBox(int class_id){
	return output_bboxes_[class_id];
}

template <typename Dtype>
void ROIOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}


#ifdef CPU_ONLY
STUB_GPU(ROIOutputLayer);
#endif



INSTANTIATE_CLASS(ROIOutputLayer);
REGISTER_LAYER_CLASS(ROIOutput);


}  // namespace caffe

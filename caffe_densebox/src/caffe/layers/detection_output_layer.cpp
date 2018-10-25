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
void DetectionOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	ROIOutputLayer<Dtype>::LayerSetUp(bottom,top);
	const DetectionOutputParameter& detection_output_param = this->layer_param_.detection_output_param();
	refine_out_of_map_bbox_ = detection_output_param.refine_out_of_map_bbox();
}
/**
 * @todo
 */
template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	int num = bottom[0]->num();
	pyramid_image_data_param_.ReadFromSerialized(*(bottom[1]), 0);
	RectBlockPacking<Dtype>&  block_packer = pyramid_image_data_param_.rect_block_packer_;
	const int heat_map_a = pyramid_image_data_param_.heat_map_a_;
	const int heat_map_b = pyramid_image_data_param_.heat_map_b_;
	const int map_height = bottom[0]->height();
	const int map_width = bottom[0]->width();
	const int map_size = map_height* map_width;
	const int scale_num = bottom[0]->channels()/this->channel_per_scale_;
	CHECK_EQ(bottom[0]->channels()%this->channel_per_scale_,0);
	CHECK_GE(this->channel_per_scale_, 5);
	const int bbox_data_dim = this->channel_per_scale_ -  5;
	CHECK_EQ(bbox_data_dim % 3, 0);
	const int n_additional_point = bbox_data_dim / 3;
	vector<Dtype* > bbox_data_ptr(bbox_data_dim,static_cast<Dtype*>(NULL));

	const int pad_h_map = block_packer.pad_h()/heat_map_a;
	const int pad_w_map = block_packer.pad_w()/heat_map_a;
//	timer_total_.Start();
	this->timer_get_bbox_.Start();
	/**
	 * Get bbox from botom[0]
	 */
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	for(int i=0; i < num; ++i){
		int block_id = pyramid_image_data_param_.GetBlockIdBy(i);
		int cur_map_start_x , cur_map_start_y;
		block_packer.GetFeatureMapStartCoordsByBlockId( block_id, heat_map_a ,
				 heat_map_b , cur_map_start_y, cur_map_start_x);
//		LOG(INFO)<<"block_id<<"<<block_id<<" has the map_start loc: ("<< cur_map_start_y<<","<<cur_map_start_x<<").";
		for(int scale_id = 0; scale_id < scale_num; ++scale_id){
			for(int class_id = 0; class_id < this->num_class_; ++class_id){
				// Get bbox candidates
				int score_channel = scale_id*this->channel_per_scale_ + class_id;
				int offset_channel = scale_id*this->channel_per_scale_ + this->num_class_ + class_id*4;
				Dtype* scores = bottom_data + bottom[0]->offset(i,score_channel  , 0,0);
				Dtype* dx1 =    bottom_data + bottom[0]->offset(i,offset_channel+0,0,0);
				Dtype* dy1 =    bottom_data + bottom[0]->offset(i,offset_channel+1,0,0);
				Dtype* dx2 =    bottom_data + bottom[0]->offset(i,offset_channel+2,0,0);
				Dtype* dy2 =    bottom_data + bottom[0]->offset(i,offset_channel+3,0,0);
				for(int bbox_data_dim_id = 0; bbox_data_dim_id < bbox_data_dim; ++bbox_data_dim_id){
					bbox_data_ptr[bbox_data_dim_id] = bottom_data + bottom[0]->offset(
							i,offset_channel+4+bbox_data_dim_id,0,0);
				}

				for(int off = 0; off< map_size; ++off){
					if(scores[off] < this->threshold_)
						continue;
					int h = off / map_width;
					int w = off % map_width ;
					if( h >=  pad_h_map && h < map_height - pad_h_map && w >= pad_w_map  && w < map_width - pad_w_map){
						int center_buffered_img_h = (h + cur_map_start_y) * heat_map_a + heat_map_b ;
						int center_buffered_img_w = (w + cur_map_start_x) * heat_map_a + heat_map_b ;
						int roi_id = block_packer.GetRoiIdByBufferedImgCoords( center_buffered_img_h, center_buffered_img_w);
						if(roi_id > -1){
			//				LOG(INFO)<<"find bbox at ("<<center_buffered_img_h<<","<<center_buffered_img_w<<"), with roi_id:"<<roi_id;

							this->all_candidate_bboxes_[class_id].push_back(BBox<Dtype>());
							int cur_class_box_size = this->all_candidate_bboxes_[class_id].size();
							BBox<Dtype>& bbox = this->all_candidate_bboxes_[class_id][cur_class_box_size -1];
							bbox.score = scores[off];
							block_packer.GetInputImgCoords(roi_id,center_buffered_img_h - dy1[off]*heat_map_a,
									center_buffered_img_w - dx1[off]*heat_map_a, bbox.y1, bbox.x1);
							block_packer.GetInputImgCoords(roi_id,center_buffered_img_h - dy2[off]*heat_map_a,
													center_buffered_img_w - dx2[off]*heat_map_a, bbox.y2, bbox.x2);

							bbox.data.resize(bbox_data_dim,0);
							for(int bbox_data_dim_id = 0; bbox_data_dim_id < n_additional_point; ++bbox_data_dim_id){
								bbox.data[bbox_data_dim_id] = bbox_data_ptr[bbox_data_dim_id][off];
								block_packer.GetInputImgCoords(roi_id,
										center_buffered_img_h -
										bbox_data_ptr[bbox_data_dim_id*2+n_additional_point+1][off]*heat_map_a,
										center_buffered_img_w -
										bbox_data_ptr[bbox_data_dim_id*2+n_additional_point  ][off]*heat_map_a,
										bbox.data[bbox_data_dim_id*2+n_additional_point+1],
										bbox.data[bbox_data_dim_id*2+n_additional_point]);
							}


							if(refine_out_of_map_bbox_){
								bbox.x1 = MAX(bbox.x1,0);
								bbox.y1 = MAX(bbox.y1,0);
								bbox.x2 = MIN(bbox.x2,pyramid_image_data_param_.img_w_);
								bbox.y2 = MIN(bbox.y2,pyramid_image_data_param_.img_h_);
							}

						}
					}
				}
			}
		}
	}
	this->time_get_bbox_ += this->timer_get_bbox_.MilliSeconds();

	/**
	 * If this is the last forward for thi image, do nms
	 */
	if(pyramid_image_data_param_.forward_iter_id_ == (pyramid_image_data_param_.forward_times_for_cur_sample_ -1)){
		for(int class_id = 0; class_id < this->num_class_; ++class_id){
			this->timer_nms_.Start();
			vector<BBox<Dtype> > & cur_box_list = this->all_candidate_bboxes_[class_id];
			vector<BBox<Dtype> > & cur_outbox_list = this->output_bboxes_[class_id];
			this->is_candidate_bbox_selected_ = caffe::nms(cur_box_list,this->nms_overlap_ratio_,this->nms_top_n_,false);
			cur_outbox_list.clear();
			for(int i=0 ; i < this->is_candidate_bbox_selected_.size(); ++i){
				if(this->is_candidate_bbox_selected_[i]){
					cur_outbox_list.push_back(cur_box_list[i]);
				}
			}
			cur_box_list.clear();
			this->time_nms_ += this->timer_nms_.MilliSeconds();
		}
		if(this->show_time_){
			LOG(INFO)<<"Time cost for detection out layer: "<<this->time_nms_+ this->time_get_bbox_<<" ms, in which get bbox cost "<<
					this->time_get_bbox_<<" ms,  and nms cost "<<this->time_nms_ <<"ms. ";
		}
		this->time_get_bbox_ = this->time_total_ = this->time_nms_ = 0;
	}

}


template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(DetectionOutputLayer);
#endif


INSTANTIATE_CLASS(DetectionOutputLayer);
REGISTER_LAYER_CLASS(DetectionOutput);
//INSTANTIATE_LAYER_GPU_FORWARD(DetectionOutputLayer);

}  // namespace caffe

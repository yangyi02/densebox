#include <string>
#include <vector>
#include <cfloat>
#include <algorithm>
#include "caffe/layers/pyramid_data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"
#include "caffe/util/io.hpp"
namespace caffe {

/**
 * @TODO
 */
template <typename Dtype>
void ROIShowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	CHECK(this->layer_param_.has_roi_show_param());
	const ROIShowParam& roi_show_param = this->layer_param_.roi_show_param();

	CHECK(roi_show_param.has_heat_map_a());
	CHECK(roi_show_param.has_heat_map_b());
	heat_map_a_ = roi_show_param.heat_map_a();
	heat_map_b_ = roi_show_param.heat_map_b();
	mean_bgr_[0] = roi_show_param.mean_b();
	mean_bgr_[1] = roi_show_param.mean_g();
	mean_bgr_[2] = roi_show_param.mean_r();
	is_input_heatmap_ = roi_show_param.is_input_heatmap();
	heatmap_threshold_ = roi_show_param.heatmap_threshold();
	img_count = 0;
	show_output_path_ = string("cache/ROIShowLayer");
	char output_path[512];
	sprintf(output_path, "%s/pic", show_output_path_.c_str());
	CreateDir(output_path);
}


template <typename Dtype>
void ROIShowLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	this->input_num_ = bottom[0]->num();
	this->input_h_ = bottom[0]->height();
	this->input_w_ = bottom[0]->width();
	has_the_fourth_blob_ = false;
	CHECK_EQ(bottom[0]->channels(),3);

	if(is_input_heatmap_ == false){
		CHECK_EQ(bottom[1]->width(),5);
		CHECK_EQ(bottom[2]->width(),4);
		CHECK_EQ(bottom[1]->num(),bottom[2]->num());

		if(bottom.size() > 3){
			has_the_fourth_blob_ = true;
			if(bottom[3]->width() == 1){
				the_fourth_blob_ROI_label_ = true;
				CHECK_EQ(bottom[1]->num(),bottom[3]->num());
			}else if(bottom[3]->width() == 6){
				the_fourth_blob_ROI_label_ = false;
			}else{
				CHECK(false)<<"The fourth bottom blob should be either ROI_label( width = 1) "<<
						"or ground_truth_BBoxes( width = 6)";
			}
		}
	}
	else{
		CHECK_EQ(bottom.size(),2);
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	}
}

template <typename Dtype>
void ROIShowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	if(is_input_heatmap_ == false){
		Show_ROIs(bottom, top);
	}else {
		Show_HeatMap(bottom, top);
	}
}

template <typename Dtype>
void ROIShowLayer<Dtype>::Show_HeatMap(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	string layer_name = this->layer_param_.name();
	const int num_img = bottom[0]->num();
	const Blob<Dtype>& input_blob = *(bottom[0]);
	char path[512];
	Blob<Dtype>& label_blob = *(bottom[1]);
	const int num_class = label_blob.channels()/5;
	CHECK_EQ(label_blob.channels()%5,0);


	Dtype* label_data = label_blob.mutable_cpu_data();

	for(int i=0; i < num_img; ++i){
		for(int class_id = 0; class_id < num_class ; ++class_id){
			cv::Mat cv_img_original = caffe::BlobImgDataToCVMat(input_blob, i, mean_bgr_[0],
				mean_bgr_[1],mean_bgr_[2]);
			int pos_count = 0;
			for (int  h = 0; h < label_blob.height(); ++h)
			{
				for(int w = 0 ; w < label_blob.width(); ++ w)
				{
					int score_channel =  class_id;
					int offset_channel =  num_class + class_id*4;
					Dtype  scores = label_data[label_blob.offset(i,score_channel  , h,w)];
					Dtype  dx1 =    label_data[label_blob.offset(i,offset_channel+0,h,w)];
					Dtype  dy1 =    label_data[label_blob.offset(i,offset_channel+1,h,w)];
					Dtype  dx2 =    label_data[label_blob.offset(i,offset_channel+2,h,w)];
					Dtype  dy2 =    label_data[label_blob.offset(i,offset_channel+3,h,w)];

					const int tmp_h = h * heat_map_a_ + heat_map_b_;
					const int tmp_w = w * heat_map_a_ + heat_map_b_;

					if(scores!= Dtype(0)  && scores > heatmap_threshold_)
					{
						pos_count++;
						cv_img_original.at<cv::Vec3b>(tmp_h, tmp_w)[0] = static_cast<uint8_t>(0 * 255 * MIN(1, MAX(0, scores)));
						cv_img_original.at<cv::Vec3b>(tmp_h, tmp_w)[1] = static_cast<uint8_t>(1 * 255 * MIN(1, MAX(0, scores)));
						cv_img_original.at<cv::Vec3b>(tmp_h, tmp_w)[2] = static_cast<uint8_t>(0 * 255 * MIN(1, MAX(0, scores)));
					}
					if(dx1 != Dtype(0)  && scores > heatmap_threshold_ )
					{
						Dtype coord[4];
						coord[0] = tmp_w- dx1 * heat_map_a_ ;
						coord[0] = MIN(cv_img_original.cols, MAX(0, coord[0]));
						coord[1] = tmp_h- dy1* heat_map_a_ ;
						coord[1] = MIN(cv_img_original.rows, MAX(0, coord[1]));
						coord[2] = tmp_w- dx2* heat_map_a_ ;
						coord[2] = MIN(cv_img_original.cols, MAX(0, coord[2]));
						coord[3] = tmp_h- dy2* heat_map_a_ ;
						coord[3] = MIN(cv_img_original.rows, MAX(0, coord[3]));
						cv::rectangle(cv_img_original, cv::Point( coord[0], coord[1]),
											cv::Point(coord[2],coord[3]), cv::Scalar(255 * 0,
													255 * 1 , 255 * 1));
						cv_img_original.at<cv::Vec3b>(tmp_h, tmp_w)[2] = 255;
					}
				}
			}

			if(pos_count <= 0 )
				continue;
			sprintf(path, "%s/pic/%s_%06d_class_%04d_roi.jpg", show_output_path_.c_str(),
					layer_name.c_str(),img_count , class_id  );
			LOG(INFO)<< "Saving ROI map: " << path <<"   with "<<pos_count << "proposals.";
			imwrite(path, cv_img_original);
		}
		img_count++;
	}


}



/**
 * @brief Show ROI
 *		  If the input is ROIs, the input has at least two blobs: blob[0] is the input images,  blob[1] contains the ROIs,
 *		  and blob[2] is the ROI_info. blob[3] is optional for ground truth bboxes or ROI_label.
 *		  If the input is heatmap, the input has two blobs: blob[0] is the input images,  blob[1] for heatmap
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
void ROIShowLayer<Dtype>::Show_ROIs(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	string layer_name = this->layer_param_.name();
	const int num_img = bottom[0]->num();
	const int num_roi = bottom[1]->num();
	if (num_roi <= 0)
		return;
	const Blob<Dtype>& input_blob = *(bottom[0]);
	char path[512];

	Dtype* in_ROIs_data = bottom[1]->mutable_cpu_data();
	Dtype* in_ROIs_info_data = bottom[2]->mutable_cpu_data();
	Dtype* in_ROI_label = NULL;
	Dtype* in_GT_BBOX = NULL;
	if(has_the_fourth_blob_ && the_fourth_blob_ROI_label_){
		in_ROI_label = bottom[3]->mutable_cpu_data();
	}
	if(has_the_fourth_blob_ && (!the_fourth_blob_ROI_label_)){
		in_GT_BBOX = bottom[3]->mutable_cpu_data();
	}
	CHECK(!(in_ROI_label != NULL && in_GT_BBOX != NULL));

	// find the num_ of class_id
	int max_class_id = -1;
	for(int j=0; j < num_roi; ++j){
		Dtype* cur_ROI_info_data = in_ROIs_info_data + j * 4;
		int class_id = cur_ROI_info_data[0]+1;
		max_class_id = max_class_id >= class_id ? max_class_id:class_id;
	}
	CHECK_GT(max_class_id,0);

	for(int i=0; i < num_img; ++i){
		for(int cur_class_id = 1; cur_class_id <= max_class_id; ++cur_class_id){
			cv::Mat cv_img_original = caffe::BlobImgDataToCVMat(input_blob, i, mean_bgr_[0],
				mean_bgr_[1],mean_bgr_[2]);
			// loop all ROIs to find ROIs in current image
			int class_count = 0;
			for(int j=0; j < num_roi; ++j){
				Dtype* cur_ROI_data = in_ROIs_data + j * 5;
				int img_id = cur_ROI_data[0];

				if(img_id != i)
					continue;
				Dtype* cur_ROI_info_data = in_ROIs_info_data + j * 4;
				int thickness = 1;
				int class_id = cur_ROI_info_data[0]+1;
				LOG(INFO)<<"In show, find class_id "<<class_id -1<<"  in roi " <<
					j<<" , but matched Class_id is "<< cur_class_id -1;
				if(class_id != cur_class_id)
				{
					continue;
				}
				if(has_the_fourth_blob_ && the_fourth_blob_ROI_label_){
					if(class_id == in_ROI_label[j]){
						class_id += 1000;
						thickness += 1;
					}
				}
				class_count++;
				cv::Point pt_1 = cv::Point(cur_ROI_data[1]*heat_map_a_ + heat_map_b_,
						cur_ROI_data[2]*heat_map_a_ + heat_map_b_ );
				cv::Point pt_2 = cv::Point(cur_ROI_data[3]*heat_map_a_ + heat_map_b_,
						cur_ROI_data[4]*heat_map_a_ + heat_map_b_ );
				cv::rectangle(cv_img_original, pt_1,pt_2,caffe::GetColorById(class_id),thickness);
			}
			// print ground truth bbox
			if(has_the_fourth_blob_ && (!the_fourth_blob_ROI_label_)){
				int gt_bbox_num = bottom[3]->num();
				for(int j=0; j < gt_bbox_num; ++j){
					Dtype* cur_GT_bbox_data = in_GT_BBOX + j * 6;
					int img_id = cur_GT_bbox_data[1];
					if(img_id != i)
						continue;
					int thickness = 2;
					int class_id = cur_GT_bbox_data[0]+1 ;
					if(class_id != cur_class_id)
					{
						continue;
					}
					class_id += 1000;
					class_count++;
					cv::Point pt_1 = cv::Point(cur_GT_bbox_data[2]*heat_map_a_ + heat_map_b_,
							cur_GT_bbox_data[3]*heat_map_a_ + heat_map_b_ );
					cv::Point pt_2 = cv::Point(cur_GT_bbox_data[4]*heat_map_a_ + heat_map_b_,
							cur_GT_bbox_data[5]*heat_map_a_ + heat_map_b_ );
					cv::rectangle(cv_img_original, pt_1,pt_2,caffe::GetColorById(class_id),thickness);
				}
			}
			if(class_count <= 0 )
				continue;
			sprintf(path, "%s/pic/%s_%06d_class_%04d_roi.jpg", show_output_path_.c_str(),
					layer_name.c_str(),img_count, cur_class_id -1 );
			LOG(INFO)<< "Saving ROI map: " << path <<"   with "<<class_count << "proposals.";
			imwrite(path, cv_img_original);
		}
		img_count++;
	}

}


template <typename Dtype>
void ROIShowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){
	Forward_cpu(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(ROIShowLayer);
#endif



INSTANTIATE_CLASS(ROIShowLayer);
REGISTER_LAYER_CLASS(ROIShow);


}  // namespace caffe

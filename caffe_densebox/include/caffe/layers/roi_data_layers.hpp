#ifndef CAFFE_ROI_DATA_LAYERS_HPP_
#define CAFFE_ROI_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"
#include "caffe/util/RectMap.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"
using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {

/**
 * @brief Convert detection feature map or ROIs to bboxes(ROIs) and bbox(ROI)_info, which are stored in top[0]
 * 		  and top[1] respectively.
 * 		  case 1: bottom[0] = feature map  then apply threshold and nms
 * 		  case 2: bottom[0] = ROIs  bottom[1] = ROI_info, then apply threshold and nms

 * 		  Each bbox is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  Each bbox_info is represented as (class_id,  w, h , score);
 *
 */
template <typename Dtype>
class ROIOutputLayer : public Layer <Dtype>{
public:
	explicit ROIOutputLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROIOutputLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){};

	virtual inline const char* type() const { return "ROIOutput"; }
	virtual inline int  MinBottomBlobs() const { return 1; }
	virtual inline int  MaxBottomBlobs() const { return 2; }
	virtual inline int  ExactNumTopBlobs() const { return 2; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

	vector< BBox<Dtype> >& GetFilteredBBox(int class_id);
	inline vector<vector< BBox<Dtype> > >& GetFilteredBBox(){
		return output_bboxes_;
	}
	inline int GetNumClass(){
		return num_class_;
	}
	inline vector<string> GetClassNames(){
		return class_names_;
	}

private:
	void GetROIsFromFeatureMap(const vector<Blob<Dtype>*>& bottom);
	void GetROIsFromROIs(const vector<Blob<Dtype>*>& bottom);
	bool is_bottom_rois_;

	vector<vector<int> > ROI_num_id_list_;
protected:
	vector<vector<BBox<Dtype> > > all_candidate_bboxes_;
	vector<bool> is_candidate_bbox_selected_;
	vector< vector< BBox<Dtype> > > output_bboxes_;

	int bbox_data_size, bbox_info_size;

	Dtype threshold_;
	bool nms_need_nms_;
	Dtype nms_overlap_ratio_;
	int nms_top_n_;
	bool nms_add_score_;

	int channel_per_scale_;
	int num_class_;
	vector<string> class_names_;

	bool show_time_;
	Dtype time_get_bbox_, time_total_, time_nms_, time_bbox_to_blob_;
	Timer timer_get_bbox_;
	Timer timer_total_;
	Timer timer_nms_;
	Timer timer_bbox_to_blob_;
};

/**
 * @brief Generate ROI labels and target_bbox_diff from ground truth BBox and given ROIs and ROI_info.
 *		  Bottom: blob[0]-> ROIs   blob[1] ->ROI_info  blob[2] -> GT_BOX   (optional )blob[3] -> predicetd_label
 *		  The output has 4 blobs, top[0] is the filtered ROI, and top[1] is the ROI_info.
 *		  top[2] and top[3] output the ROI labels and target_bbox_diff. (optional )top[4] -> predicetd_label
 *		  ROI label is a non-negative integer, with 0 indicating background.
 *		  You should be careful that class_id in ROI_info does not contain background class. So the correspondence
 *		  betreen class_id and ROI_label is : ROI_label = class_id +1
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROIDataLayer : public Layer <Dtype>{
public:
	explicit ROIDataLayer(const LayerParameter& param);

	virtual ~ROIDataLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIData"; }
	virtual inline int  MinBottomBlobs() const { return 3; }
	virtual inline int  MaxBottomBlobs() const { return 4; }
	virtual inline int  ExactNumTopBlobs() const { return 4; }
	virtual inline int  MinTopBlobs() const { return 4; }
	virtual inline int  MaxTopBlobs() const { return 5; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
protected:

	void FilterInputROIsWithIOU(const vector<Blob<Dtype>*>& bottom);
	bool BalanceROIs(const vector<Blob<Dtype>*>& bottom);
	void ShuffleIds(vector<int>& ids);
	unsigned int PrefetchRand();
	Dtype PrefetchRandFloat();


	Dtype pos_iou_ratio_, neg_iou_ratio_;
	int num_class_;
	vector<Dtype> valid_ROIs_;
	vector<Dtype> valid_ROI_info_;
	vector<Dtype> valid_ROIs_label_;
	vector<Dtype> valid_ROIs_bbox_diff_;
	vector<Dtype> valid_IOUs_;

	vector<Dtype> out_ROIs_;
	vector<Dtype> out_ROI_info_;
	vector<Dtype> out_ROIs_label_;
	vector<Dtype> out_ROIs_bbox_diff_;
	vector<Dtype> out_ROIs_predict_;

	vector<int> pos_ids_;
	vector<int> normal_neg_ids_;
	vector<int> hard_neg_ids_;
	vector<int> selected_ids_;

	int ROI_data_length_;
	int ROI_info_length_;
	int GT_data_length_;
	int predict_dim_;

	bool need_balance_;
	bool has_predicted_label_;
	Dtype hard_threshold_;
	Dtype neg_ratio_;
	Dtype hard_ratio_; /// hard_neg refers to neg samples whose IOU ratio is larger than hard_threshold_
	shared_ptr<Caffe::RNG> prefetch_rng_;
};


/**
 * @brief Convert ROI output into HeatMap
 *		  The input has 4 blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the predicted labels and  bbox_diff.
 *		  Predicted label is a class_num +1 array, with the first channel indicating background probability.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROI2HeatMapLayer : public Layer <Dtype>{
public:
	explicit ROI2HeatMapLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROI2HeatMapLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROI2HeatMap"; }
	virtual inline int  ExactNumBottomBlobs() const { return 4; }
	virtual inline int  ExactNumTopBlobs() const { return 1; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:

	int num_class_, map_h_, map_w_, map_num_;
	int ROI_data_length_, ROI_info_length_;
	ROI2HeatMapParam_LabelType label_type_;

};


/**
 * @brief Refine ROI output by ROI_diff and ROI_predicted
 *		  The input has at least three blobs, bottom[0] is the filtered ROI, and bottom[1] is the ROI_info.
 *		  bottom[2] and bottom[3] denote the bbox_diff and predicted labels.
 * 		  Each ROI is represented as: ( item_id, roi_start_w, roi_start_h, roi_end_w, roi_end_w );
 *		  And ROI_info is represented as (class_id,  w, h , score);
 * 		  Each ground truth bbox is represented as: (class_id, item_id,
 * 		  roi_start_w, roi_start_h, roi_end_w, roi_end_w);
 */
template <typename Dtype>
class ROIRefineLayer : public Layer <Dtype>{
public:
	explicit ROIRefineLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROIRefineLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIRefine"; }
	virtual inline int  MinBottomBlobs() const { return 3; }
	virtual inline int  MaxBottomBlobs() const { return 4; }
	virtual inline int  ExactNumTopBlobs() const { return 2; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:
	int ROI_data_length_, ROI_info_length_, num_class_;
	ROIRefineParam_LabelType label_type_;
	bool  has_ROI_score_;

};



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
class ROIShowLayer : public Layer <Dtype>{
public:
	explicit ROIShowLayer(const LayerParameter& param)
		  :  Layer<Dtype>(param) {}
	virtual ~ROIShowLayer(){};
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIShow"; }
	virtual inline int  MinBottomBlobs() const { return 2; }
	virtual inline int  ExactNumTopBlobs() const { return 0; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
protected:

	void Show_ROIs(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	void Show_HeatMap(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);

	int heat_map_a_, heat_map_b_;
	int input_w_, input_h_,input_num_;
	bool has_the_fourth_blob_;
	bool the_fourth_blob_ROI_label_;
	Dtype mean_bgr_[3];
	string show_output_path_;

	int img_count;
	bool is_input_heatmap_;
	Dtype heatmap_threshold_;
};




}  // namespace caffe

#endif  // CAFFE_ROI_DATA_LAYERS_HPP_

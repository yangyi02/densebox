#ifndef CAFFE_PYRAMID_DATA_LAYERS_HPP_
#define CAFFE_PYRAMID_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/roi_data_layers.hpp"
#include "caffe/util/RectMap.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/caffe_wrapper_common.hpp"
#include <stdint.h>
using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {

/**
 * @brief Packing and unpacking image blocks.
 * Designed by alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BlockPacking{
public:
	BlockPacking();
	virtual ~BlockPacking();
	virtual void SetUpParameter(const BlockPackingParameter& block_packing_param);

	virtual void ImgPacking_cpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out);
	virtual void ImgPacking_gpu(const Blob<Dtype>& blob_in, Blob<Dtype>& blob_out) ;
	void FeatureMapUnPacking_cpu(const Blob<Dtype>& blob_out, Blob<Dtype> blob_in,
			const int num_in_img ,int heat_map_a_);
	void FeatureMapUnPacking_gpu(const Blob<Dtype>& blob_out, Blob<Dtype> blob_in,
			const int num_in_img ,int heat_map_a_);
	virtual int SerializeToBlob(Blob<Dtype>& blob, int start = 0);
	virtual int ReadFromSerialized(Blob<Dtype>& blob, int start = 0);
	inline void GetFeatureMapStartCoordsByBlockId(const int block_id, const int heat_map_a,
			const int heat_map_b, int& map_start_y, int& map_start_x){
		CHECK_LT(block_id,  num_block_w_ *  num_block_h_ * buff_map_.num());
		int cur_block_id = block_id %(num_block_w_ *  num_block_h_);
		map_start_y = ((cur_block_id / num_block_w_) * block_height_ -pad_h_  )/Dtype(heat_map_a);
		map_start_x = ((cur_block_id % num_block_w_) * block_width_ -pad_w_ )/Dtype(heat_map_a);
	}

	inline void setShowTime(bool show){
		show_time_ = show;
	}
	inline int num_block_w(){
		return num_block_w_;
	}
	inline int num_block_h(){
			return num_block_h_;
	}
	inline int block_width(){
			return block_width_;
	}
	inline int block_height(){
			return block_height_;
	}
	inline int pad_h(){
		return pad_h_;
	}
	inline int pad_w(){
		return pad_w_;
	}
	inline int max_stride(){
		return max_stride_;
	}
	inline int max_block_size(){
			return max_block_size_;
		}
	inline Blob<Dtype>& buff_map(){
		return buff_map_;
	}

protected:
	/*
	 * @brief Copy blob_in intp buff_map_ for further processing.
	 */
	virtual void SetInputBuff_cpu(const Blob<Dtype>& blob_in);
	virtual void SetInputBuff_gpu(const Blob<Dtype>& blob_in){};

	virtual void SetBlockPackingInfo(const Blob<Dtype>& blob_in);
	void GetBlockingInfo1D(const int in_x, int & out_x, int& out_num);

	int max_stride_;
	int pad_h_;
	int pad_w_;
	int max_block_size_;

	int num_block_w_;
	int num_block_h_;
	int block_width_;
	int block_height_;

	bool show_time_;
	/**
	 * @warning  buff_map_ might be very large!
	 * So don't synchronize this variable to gpu
	 * memory
	 */
	Blob<Dtype> buff_map_;
};

template <typename Dtype>
class RoiRect{
public:
	RoiRect(Dtype scale = 0, Dtype start_y = 0, Dtype start_x = 0,
			Dtype height = 0, Dtype width = 0);
	~RoiRect(){};
	inline Dtype GetArea() const{
		return height_*width_;
	}
	inline Dtype GetScaledArea() const {
		return scale_*height_*scale_*width_;
	}
	inline Dtype GetScaledHeight()const {
		return scale_*height_;
	}
	inline Dtype GetScaledWidth()const{
		return scale_*width_;
	}
	inline Dtype GetScaledX(Dtype dx)const{
		return (start_x_ + dx)*scale_;
	}
	inline Dtype GetScaledY(Dtype dy)const{
		return (start_y_ +dy)*scale_;
	}
	inline Dtype GetOriX(Dtype scaled_dx)const{
		return start_x_ + scaled_dx/scale_;
	}
	inline Dtype GetOriY(Dtype scaled_dy)const{
		return start_y_ + scaled_dy/scale_;
	}
	static bool greaterScale(const RoiRect& a, const RoiRect&b) {
		return a.scale_ > b.scale_;
	}
	static bool greaterScaledArea(const RoiRect& a, const RoiRect&b) {
		return a.GetScaledArea() > b.GetScaledArea();
	}
	static bool greaterMaxScaledEdge(const RoiRect& a, const RoiRect&b) {
		return MAX(a.GetScaledHeight(),a.GetScaledWidth()) >
			MAX(b.GetScaledHeight(),b.GetScaledWidth());
	}
	friend ostream& operator << (ostream & stream,RoiRect & rect){
		stream<< "("<<rect.scale_<<","<<rect.start_y_<<","<<rect.start_y_<<","<<
				rect.height_<<","<<rect.width_<<")";
		return stream;
	}
	int SerializeToBlob(Blob<Dtype>& blob, int start = 0);
	int ReadFromSerialized(Blob<Dtype>& blob, int start = 0);
	Dtype scale_;
	Dtype start_y_;
	Dtype start_x_;
	Dtype height_;
	Dtype width_;

};

/**
 * @brief Packing and unpacking image blocks. This class enables PatchWorks
 * over scales on the input blob. Unlike the BlobkPacking class, the input
 * blob should have num() == 1.
 * Designed by alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class RectBlockPacking :public BlockPacking<Dtype>{
public:
	RectBlockPacking(){};
	virtual ~RectBlockPacking(){};
	///call this function before packing
	void setRoi(const Blob<Dtype>& blob_in,const vector<Dtype> scales, const int max_size = 200000);
	void setRoi(const Blob<Dtype>& blob_in,const pair<string, vector<Dtype> >&  cur_sample);
	virtual int SerializeToBlob(Blob<Dtype>& blob, int start = 0);
	virtual int ReadFromSerialized(Blob<Dtype>& blob, int start = 0);

	int GetRoiIdByBufferedImgCoords(const int coords_y, const int coords_x);
	inline void GetInputImgCoords(const int roi_id, const Dtype buff_img_y, const Dtype buff_img_x,
			 Dtype& input_y, Dtype& input_x){
		int left_top_x =  placedRect_[roi_id].left_top.x;
		int left_top_y =  placedRect_[roi_id].left_top.y;
		input_y = roi_[roi_id].GetOriY(buff_img_y-left_top_y);
		input_x = roi_[roi_id].GetOriX(buff_img_x-left_top_x);
	}
protected:
	virtual void SetBlockPackingInfo(const Blob<Dtype>& blob_in);
	virtual void SetInputBuff_cpu(const Blob<Dtype>& blob_in);
	virtual void SetInputBuff_gpu(const Blob<Dtype>& blob_in);
	/// serialize Rect to blob;
	int SerializeToBlob(Blob<Dtype>& blob, Rect& rect,int start = 0);
	int ReadFromSerialized(Blob<Dtype>& blob, Rect& rect,int start = 0);

	RectMap rectMap_;
	vector<RoiRect<Dtype> > roi_;
	vector<Rect> placedRect_;
	/**
	 * buffers for crop and resize
	 */
	Blob<Dtype> buff_blob_1_;
	Blob<Dtype> buff_blob_2_;

};

/**
 * @brief The same as parameters in PyramidImageDataLayer.
 * This struct is used for parameter passing.
 */
template <typename Dtype>
struct PyramidImageDataParam{

	int ReadFromSerialized(Blob<Dtype>& blob, int start = 0);
	inline int GetBlockIdBy(const int num_id){
		return forward_iter_id_*max_block_num_ + num_id;
	}
	int img_w_, img_h_;
	int heat_map_a_;
	int heat_map_b_;
	int max_block_num_;
	RectBlockPacking<Dtype> rect_block_packer_;
	int forward_times_for_cur_sample_;
	int forward_iter_id_;
};


template <typename Dtype>
class PyramidImageDataLayer :  public Layer <Dtype>,public OLD_InternalThread{
public:
	explicit PyramidImageDataLayer(const LayerParameter& param)
	  :  Layer<Dtype>(param) {
		img_from_memory_ = false;
	}
	virtual ~PyramidImageDataLayer();
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top){};

	virtual inline const char* type() const { return "PyramidImageData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }

	/*
	 * @brief top[0] contains the blocked image, and top[1] pass the
	 * PyramidImageDataParam to next layer.
	 */
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	inline pair<string, vector<Dtype> > GetCurSample(){
		return *(cur_sample_list_[used_buffered_block_id_]);
	}
	int GetTotalSampleSize(){
		return samples_provider_.GetSamplesSize();
	}
	int GetCurSampleID(){
		return cur_sample_id_;
	}
	int GetForwardTimesForCurSample(){
		return forward_times_for_cur_sample_;
	}
	int GetCurForwardIDForCurSample(){
		return forward_iter_id_;
	}

	void SetProcessMode(Caffe::Brew mode){
		this->mode_ = mode;
	}
protected:
	virtual void CreatePrefetchThread();
	virtual void JoinPrefetchThread();
	virtual void InternalThreadEntry();

	virtual void LoadSampleToImgBlob(pair<string, vector<Dtype> >& cur_sample_,
			Blob<Dtype>& img_blob_);
	virtual void SetRoiAndScale(RectBlockPacking<Dtype>& rect_block_packer_,
			pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_);

	virtual void StartProcessOneImg();

	virtual int SerializeToBlob(Blob<Dtype>& blob, int start = 0);

	void ShowImg(const vector<Blob<Dtype>*>& top);

	ImageDataSourceProvider<Dtype> samples_provider_;

	bool img_from_memory_;

	bool shuffle_;
	Dtype img_w_[2];
	Dtype img_h_[2];

	Dtype mean2_bgr_[3];
	bool is_img_pair_;


	int heat_map_a_;
	int heat_map_b_;
	Dtype mean_bgr_[3];
	Dtype scale_start_;
	Dtype scale_end_;
	Dtype scale_step_;
	bool scale_from_annotation_;
	int max_block_num_;
	int max_input_size_;


	pair<string, vector<Dtype> > cur_sample_1_;
	pair<string, vector<Dtype> > cur_sample_2_;
	vector<pair<string, vector<Dtype> >* > cur_sample_list_;

	RectBlockPacking<Dtype> rect_block_packer_1_;
	RectBlockPacking<Dtype> rect_block_packer_2_;
	vector<RectBlockPacking<Dtype>*> rect_block_packer_list_;

	Blob<Dtype> img_blob_1_;
	Blob<Dtype> img_blob_2_;
	vector<Blob<Dtype>*> img_blob_list_;

	vector<Blob<Dtype>*> buffered_block_;
	Blob<Dtype> buffered_block_1_;
	Blob<Dtype> buffered_block_2_;

	int used_buffered_block_id_;

	int forward_times_for_cur_sample_;
	int forward_iter_id_;
	int cur_sample_id_;

	string show_output_path_;
	bool pic_print_;
	bool show_time_;

	Caffe::Brew  mode_;
};

template <typename Dtype>
class PyramidImageOnlineDataLayer :  public PyramidImageDataLayer<Dtype>{
public:
	explicit PyramidImageOnlineDataLayer(const LayerParameter& param)
	  : PyramidImageDataLayer<Dtype>(param) {
		this->img_from_memory_ = true;
	}
	virtual ~PyramidImageOnlineDataLayer(){};
	virtual inline const char* type() const { return "PyramidImageOnlineData"; }

	int GetTotalSampleSize();

	void	LoadOneImgToInternalBlob(const cv::Mat& img);
	void  SetROIWithScale(const vector<ROIWithScale>& roi_scale);

protected:

	void	LoadOneImgToInternalBlob_cpu(const cv::Mat& img);
	void	LoadOneImgToInternalBlob_gpu(const cv::Mat& img);


	virtual void LoadSampleToImgBlob(pair<string, vector<Dtype> >& cur_sample_,
			Blob<Dtype>& img_blob_);
	virtual void SetRoiAndScale(RectBlockPacking<Dtype>& rect_block_packer_,
			pair<string, vector<Dtype> >& cur_sample_, Blob<Dtype>& img_blob_);
	virtual void StartProcessOneImg();


	virtual void CreatePrefetchThread(){};
	virtual void JoinPrefetchThread(){};


	Blob<uint8_t> uint8_blob_;

};



template <typename Dtype>
class DetectionOutputLayer : public ROIOutputLayer <Dtype>{
public:
	explicit DetectionOutputLayer(const LayerParameter& param)
		  :  ROIOutputLayer<Dtype>(param) {}
	virtual ~DetectionOutputLayer(){};

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "DetectionOutput"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
	virtual inline int ExactNumTopBlobs() const { return 0; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

protected:

	PyramidImageDataParam<Dtype> pyramid_image_data_param_;
	bool  refine_out_of_map_bbox_;
};





}  // namespace caffe

#endif  // CAFFE_PYRAMID_DATA_LAYERS_HPP_

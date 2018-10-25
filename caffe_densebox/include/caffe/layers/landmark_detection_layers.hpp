#ifndef CAFFE_LANDMARK_DETECTION_LAYERS_HPP_
#define CAFFE_LANDMARK_DETECTION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"
#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/layers/pyramid_data_layers.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"
using namespace caffe_fcn_data_layer;
using namespace std;
namespace caffe {



template<typename Dtype>
class FixedSizeBlockPacking{
public:
	FixedSizeBlockPacking(){
		block_h_ = block_w_ = 100;
		show_time_ = false;

	};
	FixedSizeBlockPacking(int block_h, int block_w ){
			block_h_ = block_h;
			block_w_ = block_w;
			show_time_ = false;

	};
	~FixedSizeBlockPacking(){};

	int SerializeToBlob(Blob<Dtype>& blob, int start , vector<RoiRect<Dtype> >& rois);
	int ReadFromSerialized(Blob<Dtype>& blob, int start , vector<RoiRect<Dtype> >& rois);

	void GetInputImgCoords(const RoiRect<Dtype>& roi, const Dtype buff_img_y, const Dtype buff_img_x,
			 Dtype& input_y, Dtype& input_x);

	void ImgPacking_cpu(const Blob<Dtype>& blob_in, RoiRect<Dtype>& roi, Blob<Dtype>& blob_out, int out_num_id);
	void ImgPacking_gpu(const Blob<Dtype>& blob_in, RoiRect<Dtype>& roi, Blob<Dtype>& blob_out, int out_num_id);

	inline void setShowTime(bool show){
		show_time_ = show;
	}

	inline void setBlockH(int block_h){
		block_h_ = block_h;
	}
	inline int block_h(){
		return block_h_;
	}
	inline void setBlockW(int block_w){
		block_w_ = block_w;
	}
	inline int block_w(){
		return block_w_;
	}
protected:

	Blob<Dtype> buff_blob_1_;
	Blob<Dtype> buff_blob_2_;

	int  block_h_, block_w_;
	bool show_time_;

};


template<typename Dtype>
class LandmarkDetectionDataParam{
public:

	int SerializeToBlob(Blob<Dtype>& blob, int start );
	int ReadFromSerialized(Blob<Dtype>& blob, int start );
	void SetUpParam(const LandmarkDetectionDataParameter & landmark_detection_data_param);

	int heat_map_a_, heat_map_b_;
	Dtype mean_bgr_[3];
	int  block_h_, block_w_;
	FixedSizeBlockPacking<Dtype> fixed_size_block_packer_;
	vector<RoiRect<Dtype> > rois_;

};

template<typename Dtype>
class LandmarkDetectionDataLayer : public LandmarkDetectionDataParam<Dtype>, public OLD_BasePrefetchingDataLayer<Dtype>{

public:
	explicit LandmarkDetectionDataLayer(const LayerParameter& param)
	  : OLD_BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~LandmarkDetectionDataLayer(){
		this->JoinPrefetchThread();
	};
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 2; }
	virtual inline const char* type() const { return "LandmarkDetectionData"; }
	inline vector<std::pair<std::string, vector<Dtype> > >& batch_samples(){
		return batch_samples_;
	}
	inline int GetTotalSampleSize(){
		return std::ceil(data_provider_.GetPosSampleSize());
	}
	inline int GetBatchSize(){
		return batch_size_;
	}
protected:
	virtual void InternalThreadEntry();
	void GetRoi(const int img_w, const int img_h,const vector<Dtype>& coords,
			int& lt_x, int& lt_y, int& rb_x, int& rb_y);
	void ShowImg(const Blob<Dtype>& top);

	string show_output_path_;

	LandmarkDetectionDataParam<Dtype> landmark_detection_data_param_;

	int batch_size_;
	int num_anno_points_per_instance_;
	ImageDataSourceBootstrapableProvider<Dtype> data_provider_;

	vector<RoiRect<Dtype> > prefetch_rois_;

	vector<std::pair<std::string, vector<Dtype> > > prefetch_batch_samples_;

	vector<std::pair<std::string, vector<Dtype> > > batch_samples_;



	Blob<Dtype> image_blob_;
	shared_ptr< BufferedImgReader<Dtype> > buffered_reader_;

	int roi_center_point_;
	int standard_len_point_1_, standard_len_point_2_;
	int standard_len_;
	int min_valid_standard_len_;
	bool restrict_roi_in_center_;
	bool pic_print_;

	int count_;
};




/**
 *  bottom[0]: predicted result
 *  bottom[1]: ground truth
 *  bottom[2]: input data
 */

template<typename Dtype>
class LandmarkDetectionOutputLayer : public Layer <Dtype>{
public:
	explicit LandmarkDetectionOutputLayer(const LayerParameter& param)
	  	  	  :  Layer<Dtype>(param) {}
	virtual ~LandmarkDetectionOutputLayer(){};
	virtual inline const char* type() const { return "LandmarkDetectionOutput"; }

//	virtual inline int ExactNumBottomBlobs() const { return 2; }

	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }

	virtual inline int ExactNumTopBlobs() const { return 0; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){};

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
	inline vector< vector< ScorePoint<Dtype> > >& GetResultPoints(){
		return output_points_;
	}
	inline vector<RoiRect<Dtype> >& GetOutputRois(){
		return landmark_detection_data_param_.rois_;
	}
protected:

	void GetUnFilteredPointsForOneFaceInBlob(Blob<Dtype>& blob,int num_id,vector<vector<ScorePoint<Dtype> > >& res);
	void FilterPointsForOneFace(vector<vector<ScorePoint<Dtype> > >& res, vector<ScorePoint<Dtype> > & dst,
			 Dtype center_x, Dtype center_y);
	void BlockCoordsToImgCoords(const RoiRect<Dtype>& roi, vector<ScorePoint<Dtype> >& filtered_points);

	void ShowImg(const Blob<Dtype>& blob, vector<vector<ScorePoint<Dtype> > >&  all_filtered_face_points);
	void ShowImg(const Blob<Dtype>& blob, const int num_id,vector<vector<ScorePoint<Dtype> > >&  un_filtered_face_points);
//	void BlobAndResultToCvMat(const Blob<Dtype>& blob, const int num_id,vector<ScorePoint<Dtype> >& face_points_, cv::Mat& mat);




	vector< vector< ScorePoint<Dtype> > > output_points_;



	LandmarkDetectionDataParam<Dtype> landmark_detection_data_param_;

	Dtype threshold_;
	int channel_per_scale_;
	int channel_per_point_;
	int num_point_;
	vector<int> line_point_pairs_;

	bool nms_need_nms_;
	Dtype nms_dist_threshold_;
	int nms_top_n_;
	bool nms_add_score_;

	bool pic_print_;
	string show_output_path_;

	int count_;
};


}  // namespace caffe

#endif  // CAFFE_PYRAMID_DATA_LAYERS_HPP_

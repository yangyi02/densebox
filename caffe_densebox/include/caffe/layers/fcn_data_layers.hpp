#ifndef CAFFE_FCN_DATA_LAYERS_HPP_
#define CAFFE_FCN_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/caffe_fcn_data_layer.pb.h"
#include "caffe/layers/base_data_layer.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "caffe/util/buffered_reader.hpp"
#include "boost/threadpool.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_others.hpp"


namespace caffe {

using namespace caffe_fcn_data_layer;
using namespace std;

template <typename Dtype>
class IImageDataSourceProvider{
public:
	IImageDataSourceProvider(bool need_shuffle = false);
	virtual ~IImageDataSourceProvider(){};
	virtual  std::pair<std::string, vector<Dtype> >   GetOneSample() = 0;

	inline virtual void SetShuffleFlag(bool need_shuffle){
		this->shuffle_ = need_shuffle;
	}
	/**
	 * ReadSamplesFromFile: This function reads annotations from <code>filename</code>,
	 * and store all samples.
	 */
	virtual bool ReadSamplesFromFile(const string & filename, const string & folder,
			int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 ) = 0;
	virtual int	 GetSamplesSize()= 0;
	virtual void ShuffleImages()= 0;
	static vector<Dtype> SwapInstance(vector<Dtype>& coords, int num_anno_points_per_instance,
							int id1, int id2);


protected:
	shared_ptr<Caffe::RNG> prefetch_rng_;
	bool shuffle_;
};

/**
 * @brief Provides samples which are forwarded to DataLayer . Designed by  alan
 *
 */
template<typename Dtype>
class ImageDataSourceProvider : virtual public IImageDataSourceProvider<Dtype>
{
	public:
		ImageDataSourceProvider(bool need_shuffle = false);
		virtual ~ImageDataSourceProvider();
		virtual  std::pair<std::string, vector<Dtype> >  GetOneSample();
		/**
		 * ReadSamplesFromFile: This function reads annotations from <code>filename</code>,
		 * and store all samples.
		 */
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder,
				int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 );
		inline void SetLineId(int id){
			lines_id_ = MAX(0,MIN(id,samples_.size()-1)) ;
		}
		inline virtual int	 GetSamplesSize(){
			return this->samples_.size();
		}
		inline virtual void ShuffleImages(){
			caffe::rng_t* prefetch_rng =
				        static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());
			shuffle(shuffle_idxs_.begin(), shuffle_idxs_.end(), prefetch_rng);
		}
		void PushBackSample(const pair< string, vector<Dtype> > & cur_sample);
		inline static string GetSampleName(const pair< string, vector<Dtype> > & cur_sample){
			std::vector<std::string> splited_name= std_split(cur_sample.first,"/");
			return splited_name[splited_name.size()-1];
		}
		static string GetSampleName(const pair< string, vector<Dtype> > & cur_sample, int prefix_level){
			CHECK_GE(prefix_level,0);
			std::vector<std::string> splited_name= std_split(cur_sample.first,"/");
			string out = splited_name[splited_name.size()-1 - prefix_level];
			for(int i=splited_name.size() - prefix_level; i < splited_name.size(); ++i){
				out += "/" +splited_name[i];
			}
			return out;
		}
		static string GetSampleFolder(const pair< string, vector<Dtype> > & cur_sample,  string concat_symbol = "/" );

	protected:

		/** store each samples (one line in the annotation) */
		vector<std::pair<std::string, vector<Dtype> > > samples_;
		vector<int> shuffle_idxs_;
		int lines_id_;

};



template<typename Dtype>
class ImageDataROISourceProvider : virtual public IImageDataSourceProvider<Dtype>
{
	public:
		ImageDataROISourceProvider(bool need_shuffle = false);
		virtual ~ImageDataROISourceProvider();
		virtual std::pair<std::string, vector<Dtype> >  GetOneSample();

		/**
		 * ReadSamplesFromFile: This function reads annotations from <code>filename</code>,
		 * and store all samples.
		 */
		virtual bool ReadSamplesFromFile(const string & file_list_name, const string & folder,
				int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 );

		inline void SetLineId(int id){
			lines_id_ = MAX(0,MIN(id,shuffle_idxs_.size()-1)) ;
		}
		inline virtual int	 GetSamplesSize(){
			return this->shuffle_idxs_.size();
		}
		inline virtual void ShuffleImages(){
			caffe::rng_t* prefetch_rng =
				        static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());
			shuffle(shuffle_idxs_.begin(), shuffle_idxs_.end(), prefetch_rng);
		}
		void PushBackAnnoSample(const pair< string, vector<Dtype> > & cur_sample,int num_anno_points_per_instance);
		void PushBackROISample(const pair< string, vector<Dtype> > & cur_sample);
		inline void SetROIFileName(const string& f_name){
			roi_filename_ = f_name;
		}
		static const int num_roi_points_per_instance_ = 3;
	protected:
		bool ReadAnnoOrROISamplesFromFile(const string & file_list_name, const string & folder,
						int num_anno_points_per_instance, bool is_roi_filelist  );
		inline int FindAnnoSampleIdxByName(const string& name){
			map<string,int>::iterator it = img_name_map_.find(name);
			if(it != img_name_map_.end()){
				return it->second;
			}
			return -1;
		}

		/** store each samples (one line in the annotation) */
		vector<std::pair<std::string, vector<Dtype> > > anno_samples_;

		string roi_filename_;
		/**
		 * roi_samples_: Each roi_entity is a 3-point-data, which is (x1,y1,x2,y2, x_center, y_center, label1, label2),
		 * where x1, y1, x2, y2, x_center and y_center are used to normalize patch in ImageReader.
		 */

		vector<std::pair<int, vector<Dtype> > >  roi_samples_;
		map<string,int> img_name_map_;
		vector<int> shuffle_idxs_;
		int lines_id_;


};


template<typename Dtype>
class ImageDataSourceMultiClassProvider :  virtual public IImageDataSourceProvider<Dtype>
{
	public:
		ImageDataSourceMultiClassProvider(bool need_shuffle = false);
		virtual ~ImageDataSourceMultiClassProvider();
		virtual std::pair<std::string, vector<Dtype> >   GetOneSample();
		/**
		 * ReadSamplesFromFile: This function reads annotations from <code>filename</code>,
		 * and store all samples.
		 */
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder,
				int num_anno_points_per_instance, bool no_expand = false, int class_flag_id = -1  );
		void PushBackSample(const pair< string, vector<Dtype> > & cur_sample,
				int num_anno_points_per_instance,int class_flag_id );
		virtual int	 GetSamplesSize();
		virtual void ShuffleImages();

	protected:

		/** class_id start from 0. Background  class is -1.  */
		vector<int> class_ids_;
		vector<ImageDataSourceProvider<Dtype> > image_data_providers_;

		int lines_class_id_;
		vector<int> shuffle_class_idxs_;
};

/**
 * @brief A special case of ImageDataSourceProvider for hard negative sample.
 *
 */
template<typename Dtype>
class ImageDataHardNegSourceProvider:public ImageDataSourceProvider<Dtype>
{
	public:
		ImageDataHardNegSourceProvider(bool need_shuffle = false);
		virtual ~ImageDataHardNegSourceProvider();
		bool ReadHardSamplesFromFile(const string & filename,const string & neg_img_folder,
				int input_height, int input_width);
		virtual bool ReadSamplesFromFile(const string & filename, const string & folder,
				int num_anno_points_per_instance, bool no_expand = false,int class_flag_id = -1 ){
			return false;
		}
		void SetUpHardNegParam(const FCNImageDataSourceParameter & fcn_img_data_source_param);
	protected:
		Dtype bootstrap_std_length_;
		FCNImageDataSourceParameter_STDLengthType stdLengthType;
};

/**
 * @enum  ImageDataSourceSampleType::POSITIVE Sample that contains at least one positive instance
 * @enum  ImageDataSourceSampleType::ALL_NEGATIVE Sample that contains no positive instance
 * @enum  ImageDataSourceSampleType::HARD_NEGATIVE Sample that contains hard negative instance
 */
enum ImageDataSourceSampleType { SOURCE_TYPE_POSITIVE, SOURCE_TYPE_ALL_NEGATIVE, SOURCE_TYPE_HARD_NEGATIVE,
	SOURCE_TYPE_POSITIVE_WITH_ROI};

/**
 * @brief A source container that feeds instances to DataLayer.	It contains 3 types of samples
 * 		(positive, all_negative, hard_negative). This container could update hard negative samples
 * 		on the fly.
 *
 * @todo Unfinished function for bootstrap.
 */
template<typename Dtype>
class ImageDataSourceBootstrapableProvider
{
	public:
		ImageDataSourceBootstrapableProvider();
		~ImageDataSourceBootstrapableProvider();
		virtual void SetUpParameter(const FCNImageDataParameter & fcn_image_data_param)  ;
		void SetUpParameter(const FCNImageDataSourceParameter & fcn_img_data_source_param);
		/**
		 * FetchBatchSamples:  fetch all samples needed in the current batch iterations,
		 * and store them into <code>cur_batch_samples_</code> .
		 */
		void FetchBatchSamples();
		std::pair<std::string, vector<Dtype> > &  GetMutableSampleInBatchAt(int id);
		ImageDataSourceSampleType& GetMutableSampleTypeInBatchAt(int id);
		bool ReadHardSamplesForBootstrap(const string& detection_result_file, const string & neg_img_folder,
				int input_height, int input_width );

		void ReadPosAndNegSamplesFromFiles(const FCNImageDataSourceParameter & fcn_img_data_source_param,
				int num_anno_points_per_instance, int class_flag_id = -1);
		int GetTestIterations();
		inline int GetBatchSize(){
			return batch_size_;
		}
		inline int GetPosSampleSize(){
			return pos_samples_ptr->GetSamplesSize();
		}
		inline void SetNegRatio(Dtype neg_ratio){
			neg_ratio_ = neg_ratio;
		}
		inline string pos_img_folder(){
			return pos_img_folder_;
		}
		inline vector<std::pair<std::string, vector<Dtype> > > cur_batch_samples(){
			return cur_batch_samples_;
		}
	protected:

		/**
		 * FetchSamplesTypeInBatch: get the sample types in the current batch iteration. The number of
		 * different type of samples is calculated according to <code>neg_ratio_</code> and
		 * <code>bootstrap_hard_ratio_</code> .
		 */
		void FetchSamplesTypeInBatch();

		bool has_roi_file_;

		shared_ptr<IImageDataSourceProvider<Dtype> > pos_samples_ptr;
		ImageDataSourceProvider<Dtype> all_neg_samples;
		ImageDataHardNegSourceProvider<Dtype> hard_neg_samples;
		Dtype neg_ratio_;
		Dtype bootstrap_hard_ratio_;
		int batch_size_;
		bool shuffle_;
		int batch_pos_count_;
		int batch_neg_count_;
		int batch_hard_neg_count_;
		vector<ImageDataSourceSampleType> cur_batch_sample_type_ ;
		vector<std::pair<std::string, vector<Dtype> > > cur_batch_samples_;
		bool multi_class_sample_balance_;
		bool no_expand_pos_anno_;
		string pos_img_folder_;
		string neg_img_folder_;
};


/**
 * @brief Provides base for all kinds of input processing and label generating.
 * It reads param form FCNImageDataCommonParameter.  Designed by  alan
 * - heat_map_a_ heat_map_b_. The coordinate relationship between the input and output has
 * the following mapping: x_in = x_out * heat_map_a_ + b.
 *
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
enum ImageDataAnnoType { ANNO_POSITIVE, ANNO_IGNORE, ANNO_NEGATIVE };

template <typename Dtype>
class IImageDataProcessor  {
	public:
		IImageDataProcessor(){
			total_channel_need_=0;
		}
		virtual ~IImageDataProcessor(){} ;
		virtual void SetUpParameter(const FCNImageDataParameter & fcn_image_data_param);
		inline vector<Dtype> GetScaleBase(){
			return scale_base_;
		}
		vector<ImageDataAnnoType> GetAnnoTypeForAllScaleBase( Dtype scale);
	protected:
		void	SetUpParameter(const FCNImageDataCommonParameter & fcn_img_data_common_param);
		/**
		 * @return Return the total channel number being used.
		 */
		virtual int SetUpChannelInfo( const int channel_base_offset = 0){
			return total_channel_need_;
		}
		string GetPrintSampleName(const pair< string, vector<Dtype> > & cur_sample,
			const ImageDataSourceSampleType sample_type);
		void SetScaleSamplingWeight();
		int GetWeighedScaleIdByCDF(Dtype point);
		inline bool CheckValidIndexInRange(cv::Mat & cvmat,int tmp_h, int tmp_w){
			return !(tmp_h < 0 || tmp_w < 0 || tmp_h >= cvmat.rows || tmp_w >= cvmat.cols) ;
		}

		int input_height_;
		int input_width_;
		int heat_map_a_;
		int heat_map_b_;
		int out_height_;
		int out_width_;
		bool single_thread_;
		int total_channel_need_;
		int num_anno_points_per_instance_;

		/**
		 * scale_base_: For multi-scale label output. It decides how many sets of label
		 * need to generate. For example, if scale_base_= [1,2], then the output labels should
		 * have one half for scale == 1, and another half for scale == 2. Suppose a positive instance
		 * with scale == k, for each scale_base[i], if k is within the range of positive bounder,
		 * this instance should have positive labels in the ground truth of scale_base[i],
		 * or ignore labels if still within the range of ignore bounder. Instance whose scale is
		 * outside the ignore bounder should be treated as negative instance.
		 */
		vector<Dtype> scale_base_;
		vector<Dtype> scale_sampling_weight_;
		FCNImageDataCommonParameter_ScaleChooseStrategy scale_choose_stragety_;

		Dtype scale_positive_upper_bounder_; ///< the upper bounder of scale for a positive instance near one scale_base
		Dtype scale_positive_lower_bounder_; ///< the lower bounder of scale for a positive instance near one scale_base
		Dtype scale_ignore_upper_bounder_; ///< the upper bounder of scale for a positive instance to be ignored
		Dtype scale_ignore_lower_bounder_; ///< the lower bounder of scale for a positive instance to be ignored

		bool pic_print_; ///< Flag for debug. Print the data as well as the label fed to DataLayer in the form of images.
		bool label_print_;///< Flag for debug. Print the data as well as the label fed to DataLayer in the form of text.
		string show_output_path_;

		int PIC_MARGIN;
};

/**
 * @brief Reads and transforms images from data providers. Methods for data augmentations: translation, rotation, scaling.
 *	CV_8UC3 for cvmat
 * @todo Currently methods for bootstrap is unimplemented.
 *
 */
template <typename Dtype>
class IImageDataReader : virtual public IImageDataProcessor<Dtype>{
	public:
		IImageDataReader() ;
		virtual ~IImageDataReader();
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
		/**
		 *  @brief Reads and transforms one image and refine its annotations.
		 *
		 *	@param img_cv_ptr  For visualization.The cv::Mat pointer of processed image.
		 *	@param img_ori_ptr For visualization.The cv::Mat pointer of input image.
		 *	@param is_keypoint_transform_ignored The binary array indicates which bits in annotations should be
		 *	unchanged during transformation. The length of is_keypoint_transform_ignored should be the same as the
		 *	length of annotations for this sample.
		 *	@param sample_type  POSITIVE or  ALL_NEGATIVE or HARD_NEGATIVE
		 *	@param scale_base_id Indicate which scale in scale_base_ should be activated.
		 */
		bool virtual ReadImgAndTransform(int item_id,Blob<Dtype>& prefetch_data,vector<cv::Mat*>  img_cv_ptrs, vector<cv::Mat*> img_ori_ptrs,
					pair< string, vector<Dtype> > & mutable_sample, vector<bool>& is_keypoint_transform_ignored,
					ImageDataSourceSampleType sample_type,
					Phase cur_phase,int scale_base_id = 0);
		void virtual PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr,
				const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type,
				const Dtype base_scale, const Blob<Dtype>& prefetch_data,string name_postfix = "");
		/**
		 * @brief Return the scale of instance w.r.t the normalized scale after transformation.
		 */
		inline Dtype GetRefinedBaseScale(int item_id){
			return this->standard_scales_[item_id] * this->random_scales_[item_id];
		}
	protected:
		void SetUpParameter(const FCNImageDataReaderParameter& fcn_img_data_reader_param);

		void GetResizeROIRange(int& lt_x, int& lt_y, int& rb_x, int& rb_y, int item_id,
				const int ori_cv_mat_cols,  const int ori_cv_mat_rows,
				const ImageDataSourceSampleType sample_type,
				const vector<Dtype>& coords,bool & is_neg,int scale_base_id);

		void SetResizeScale(int item_id, cv::Mat & cv_img_original_no_scaled,
				const ImageDataSourceSampleType sample_type,
				const vector<Dtype>& coords,bool & is_neg,int scale_base_id);
		void SetCropAndPad(int item_id,const cv::Mat & cv_img_original,cv::Mat & cv_img,bool is_neg);

		void RefineCoords(int item_id,vector<Dtype>& coords,  vector<bool>& is_keypoint_ignored,
				const ImageDataSourceSampleType sample_type);
		void Rotate(cv::Mat & cv_img,vector<Dtype>& coords, const vector<bool>& is_keypoint_ignored);
		void SetProcessParam(int item_id, Phase cur_phase);

		inline unsigned int PrefetchRand(){
			caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
			return (*prefetch_rng)();
		}
		inline Dtype PrefetchRandFloat(){
			caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
			return  (*prefetch_rng)() / static_cast<Dtype>(prefetch_rng->max());
		}

		Phase cur_phase_;
		Dtype scale_lower_limit_; ///< the lower bounder for scaling augmentation
		Dtype scale_upper_limit_; ///< the upper bounder for scaling augmentation
		int standard_len_point_1_; ///< point for scale normalization
		int standard_len_point_2_; ///< point for scale normalization

		/**
		 *  normalize the scale of input image so that the distance of standard_len_point_1_ and
		 *  standard_len_point_2_ is standard_len_.
		 */
		int standard_len_;
		int min_valid_standard_len_;  ///< if the origional standard_len is less than it, ignore this sample.
		int roi_center_point_; ///< the point id of the roi center.
		int rand_x_perturb_;
		int rand_y_perturb_;
		vector<pair<int, int> >crop_begs_;
		vector<pair<int, int> >paddings_;

		/**
		 *  the following two variables are used in IImageDataReader
		 */
		vector<Dtype> standard_scales_;
		vector<Dtype> random_scales_;
		vector<Dtype> sample_scales_;

		vector<Dtype> center_x_;
		vector<Dtype> center_y_;
		vector<Dtype> lt_x_;
		vector<Dtype> lt_y_;
		Dtype random_rotate_degree_;
		Dtype mean_bgr_[3];
		shared_ptr<Caffe::RNG> prefetch_rng_;

		Dtype coord_jitter_; ///< Set coords to coods + coord_jitter_*standard_len_*rand(0-1)
		Dtype random_roi_prob_; ///< The probability of random set the roi in the image


		boost::shared_mutex mutex_;

		bool restrict_roi_in_center_;
};




//
/**
 * @brief Reads and transforms pair images from data providers. Methods for data augmentations: translation, rotation, scaling.
 *
 *
 * @todo Currently methods for bootstrap is unimplemented.
 *
 */
template <typename Dtype>
class IImageBufferedDataReader : virtual public IImageDataReader<Dtype>{
	public:
		IImageBufferedDataReader() ;
		virtual ~IImageBufferedDataReader();
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
		/**
		 *  @brief Reads and transforms one image and refine its annotations.
		 *
		 *	@param img_cv_ptr  The cv::Mat pointer of processed image.
		 *	@param img_ori_ptr The cv::Mat pointer of input image.
		 *	@param is_keypoint_transform_ignored The binary array indicates which bits in annotations should be
		 *	unchanged during transformation. The length of is_keypoint_transform_ignored should be the same as the
		 *	length of annotations for this sample.
		 *	@param sample_type  POSITIVE or  ALL_NEGATIVE or HARD_NEGATIVE
		 *	@param scale_base_id Indicate which scale in scale_base_ should be activated.
		 */
		bool virtual ReadImgAndTransform(int item_id,Blob<Dtype>& prefetch_data,vector<cv::Mat*>  img_cv_ptrs, vector<cv::Mat*> img_ori_ptrs,
					pair< string, vector<Dtype> > & mutable_sample, vector<bool>& is_keypoint_transform_ignored,
					ImageDataSourceSampleType sample_type,
					Phase cur_phase,int scale_base_id = 0);
		/**
		 * @brief Return the scale of instance w.r.t the normalized scale after transformation.
		 */

	protected:
		void SetUpParameter(const FCNImageDataReaderParameter& fcn_img_data_reader_param);
		void SetCropAndPad(int item_id,const Blob<Dtype> & src_img_blob,Blob<Dtype> & dst_img_blob,bool is_neg);
		void SetResizeScale(int item_id, const Blob<Dtype> & img_pair_blob,Blob<Dtype> & dst_img_blob,
				const ImageDataSourceSampleType sample_type,
				const vector<Dtype>& coords,bool & is_neg,int scale_base_id);
		void SetProcessParam(int item_id, Phase cur_phase);

		Dtype mean2_bgr_[3];

		int buffer_size_ ;
		bool is_img_pair_;
		bool is_video_img_;
		bool use_gpu_;
		bool need_buffer_;
		string img_pair_postfix_;
		std::vector<shared_ptr<Blob<Dtype> > > buff_blob_1_;
		std::vector<shared_ptr<Blob<Dtype> > > buff_blob_2_;

		shared_ptr< BufferedImgReader<Dtype> > buffered_reader_;
};




template <typename Dtype>
class IImageDataBoxNorm : virtual public IImageDataProcessor<Dtype>{
	public:
		IImageDataBoxNorm() ;
		~IImageDataBoxNorm()  ;
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

		vector<Dtype> GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance,
				int num_points_per_instance, vector<int>& bbox_point_ids);
		vector<ImageDataAnnoType> GetAnnoTypeForAllScaleBase( Dtype scale);
	private:
		void SetUpParameter(const FCNImageDataDetectionBoxParameter& fcn_img_data_detection_box_param){};
		void SetUpParameter(const FCNImageDataBoxNormParameter& fcn_img_data_box_norm_param){};

		Dtype bbox_height_;
		Dtype bbox_width_;
		FCNImageDataBoxNormParameter_BBoxSizeNormType bbox_size_norm_type_;
};

/**
 * @brief Generate ground truth related to detection task.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class IImageDataDetectionBox :virtual public IImageDataProcessor<Dtype> {
	public:
		IImageDataDetectionBox() ;
		~IImageDataDetectionBox(){} ;
		virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
		void GenerateDetectionMap(int item_id, const vector<Dtype> & coords_of_all_instance,
			  Blob<Dtype>& prefetch_label,int used_scale_base_id) ;

		void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr,
				const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type,
				const Dtype scale, const Blob<Dtype>& prefetch_label);
		void PrintLabel(int item_id,const string & output_path,const pair< string, vector<Dtype> > & cur_sample,
			  const ImageDataSourceSampleType sample_type, const Dtype scale, const Blob<Dtype>& prefetch_label);
		void ShowDataAndPredictedLabel(const string & output_path,const string & img_name, const Blob<Dtype>& data,
				const int sampleID,const Dtype* mean_bgr,const Blob<Dtype>& label,const Blob<Dtype>& predicted, Dtype threshold);
		void GTBBoxesToBlob(Blob<Dtype>& prefetch_bbox); ///<  thread unsafe function
		void PrintBBoxes(const Blob<Dtype>& prefetch_bbox);
		inline int GetDetectionBaseChannelOffset(){
			return channel_detection_base_channel_offset_;
		}
		inline int GetDetectionLabelChannelOffset(){
			return channel_detection_label_channel_offset_;
		}
		inline int GetDetectionDiffChannelOffset(){
			return channel_point_diff_from_center_channel_offset_;
		}
	protected:
		void SetUpParameter(const FCNImageDataDetectionBoxParameter& fcn_img_data_detection_box_param);
		virtual int SetUpChannelInfo( const int channel_base_offset = 0);
		vector<ImageDataAnnoType> GetAnnoTypeForAllScaleBase( Dtype scale);
		void GenerateDetectionMapForOneInstance(int item_id, const vector<Dtype> & coords_of_one_instance,
			  const Dtype scale, const vector<ImageDataAnnoType> anno_type, Blob<Dtype>& prefetch_label,
			  int used_scale_base_id);
		void LabelToVisualizedCVMat(const Blob<Dtype>& label, const int class_id,cv::Mat& out_probs, cv::Mat& ignore_out_probs,
				int item_id, int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
				bool need_regression = true,Dtype heatmap_a = 1, Dtype heatmap_b =  0);

		void PointLabelToVisualizedCVMat(const Blob<Dtype>& label, const int class_id, int point_id,cv::Mat& out_probs,
				cv::Mat& ignore_out_probs, int item_id, int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
				bool need_regression = true,Dtype heatmap_a = 1, Dtype heatmap_b =  0);

		bool IsLabelMapAllZero(const Blob<Dtype>& label, const int class_id,int item_id, int scale_base_id);

		vector<Dtype> GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance);

		int channel_detection_base_channel_offset_;
		int channel_detection_label_channel_offset_;
		int channel_detection_ignore_label_channel_offset_;
		int channel_detection_diff_channel_offset_;
		int channel_point_diff_from_center_channel_offset_;
		int channel_detection_channel_all_need_;

		int channel_point_score_channel_offset_;
		int channel_point_ignore_channel_offset_;
		int channel_point_diff_channel_offset_;

		int total_class_num_;
		int class_flag_id_;
		bool loc_regress_on_ignore_;
		int ignore_class_flag_id_;

		int min_output_pos_radius_; ///< the minimum radius of positive instance in the ground truth
		int ignore_margin_;
		Dtype bbox_height_;
		Dtype bbox_width_;
		vector<int> bbox_point_id_; ///< the left top and right bottom of bbox.
		/**
		 * the radius of positive region is calculated as bbox_valid_dist_ratio_ * bbox_height_  or
		 * bbox_valid_dist_ratio_ * bbox_width_
		 */
		Dtype bbox_valid_dist_ratio_;
		bool need_detection_loc_diff_; ///< Flag which indicate whether to generate bbox regression
		/**
		 * the radius of bbox regression region is calculated as bbox_loc_diff_valid_dist_ratio_ * bbox_height_
		 * or bbox_valid_dist_ratio_ * bbox_width_
		 */
		Dtype bbox_loc_diff_valid_dist_ratio_;

		bool need_point_diff_from_center_; ///< Flag which indicate whether to regress the location of landmark.
		bool has_point_ignore_flag_diff_from_center_;
		vector<int> point_id_for_point_diff_from_center_; /// stores the point id for location regression.
		vector<int> point_ignore_flag_id_for_point_diff_from_center_; /// stores the point id for location regression.
		FCNImageDataDetectionBoxParameter_BBoxSizeNormType bbox_size_norm_type_;


		/**
		 * Each bbox is represented as: (class_id, item_id,roi_start_w, roi_start_h, roi_end_w, roi_end_w)
		 */
		vector<vector<Dtype> > gt_bboxes_;

		boost::shared_mutex bbox_mutex_;
		bool bbox_print_;

};


/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class IImageDataKeyPoint :virtual public IImageDataProcessor<Dtype> {
 public:

  IImageDataKeyPoint()  ;
  ~IImageDataKeyPoint() ;
  virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
  void GenerateKeyPointHeatMap(int item_id, vector<Dtype> & coords,  Blob<Dtype>& prefetch_label,
		  int used_scale_base_id);
  void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr,
  		const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type,
  		const Dtype scale, const Blob<Dtype>& prefetch_label);
  void ShowDataAndPredictedLabel(const string & output_path,const string & img_name, const Blob<Dtype>& data,
		const int sampleID,const Dtype* mean_bgr,const Blob<Dtype>& label,const Blob<Dtype>& predicted, Dtype threshold);
 protected:
  void SetUpParameter(const FCNImageDataKeyPointParameter& fcn_img_data_keypoint_param);
  virtual int SetUpChannelInfo( const int channel_base_offset = 0);
  void GenerateKeyPointHeatMapForOneInstance(int item_id, const vector<Dtype> & coords_of_one_instance,
		  const Dtype scale, const vector<ImageDataAnnoType> anno_type,Blob<Dtype>& prefetch_label);
  Dtype GetScaleForCurInstance(const vector<Dtype> & coords_of_one_instance);
  vector<Dtype> GetScalesOfAllInstances(const vector<Dtype> & coords_of_all_instance);

  void FilterAnnoTypeByObjCenter(vector<ImageDataAnnoType>& anno_type, vector<Dtype>& cur_coords);

  void LabelToVisualizedCVMat(const Blob<Dtype>& label, const int valid_point_id,cv::Mat& out_probs, cv::Mat& ignore_out_probs,
		int item_id, int scale_base_id, Dtype* color_channel_weight, Dtype threshold,
		bool need_regression = true,Dtype heatmap_a = 1, Dtype heatmap_b =  0);

  int channel_point_base_offset_;
  int channel_point_valid_keypoint_channel_offset_;
  int channel_point_loc_diff_offset_;
  int channel_point_ignore_point_channel_offset_;
  int channel_point_attribute_point_channel_offset_;
  int channel_point_channel_all_need_;


  vector<int> used_key_point_idxs_;
  vector<bool> key_point_valid_flag_;
  vector<int> used_key_point_channel_offset_;

  vector<int> used_attribute_point_idxs_;
  vector<int> used_attribute_point_channel_offset_;

  int key_point_valid_distance_;
  int key_point_min_out_valid_len_;

  int ignore_flag_radius_;
  /**
   * Note: ignore_key_point_idxs_ should be the same size as used_key_point_idxs_.
   * Each ignore_key_point_idxs_[i] is the corresponding ignore flagged version of used_key_point_idxs_[i]
   */
  vector<int> ignore_key_point_flag_idxs_;
  vector<int> get_ignore_flag_idx_by_point_idx_;
  vector<bool> attribute_point_flag_idxs_;
  vector<int> get_attribute_idx_by_point_idx_;

  bool need_point_loc_diff_;
  int  key_point_loc_diff_radius_;

  int key_point_standard_point_id1_;
  int key_point_standard_point_id2_;
  int key_point_standard_length_;
  int object_center_point_id_;
};

/**
 * @brief Provides data to the Net from image files.By alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class IImageDataIgnoreBox : virtual public IImageDataProcessor<Dtype> {
 public:

  IImageDataIgnoreBox(){} ;
  ~IImageDataIgnoreBox(){};
  virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);

 protected:
  void SetUpParameter(const FCNImageDataIgnoreBoxParameter& fcn_img_data_ignore_box_param);
  virtual int SetUpChannelInfo( const int channel_base_offset = 0);
  virtual void GenerateIgnoreBoxMap(int item_id, vector<float> & coords,vector<float> & box_scale,
		  const LayerParameter& param)  ;
  void PrintPic(int item_id, const string & output_path, cv::Mat* img_cv_ptr, cv::Mat* img_ori_ptr,
  			const pair< string, vector<Dtype> > & cur_sample,const ImageDataSourceSampleType sample_type,
  			const Dtype scale, const Blob<Dtype>& prefetch_label);

  int channel_ignore_box_base_offset_;
  int ignore_box_flag_id_;
  vector<int> ignore_box_point_id_;

};

/**
 * @brief Provides data to the Net from image files.By alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit BaseImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~BaseImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:

  int   batch_size_;
  bool  single_thread_;
  bool  is_data_in_gpu_;
  int   thread_num_;
//  vector<boost::thread * > img_process_threads_;
  boost::threadpool::pool thread_pool_;


  bool need_prefetch_bbox_;

//  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch);

  virtual void BatchSetUp(Batch<Dtype>* batch){};
  virtual void BatchFinalize(Batch<Dtype>* batch){};
  virtual void ProcessImg(Batch<Dtype>* batch,int item_id){};

  vector<std::pair<std::string, vector<Dtype> > > last_batch_samples_;
};


/**
 * @brief Provides data to the Net from image files.By alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class FCNImageDataLayer : public BaseImageDataLayer<Dtype>,public IImageDataIgnoreBox<Dtype>,
			public IImageDataKeyPoint<Dtype>, public IImageDataDetectionBox<Dtype>,
			public IImageBufferedDataReader<Dtype>
{
 public:
	explicit FCNImageDataLayer(const LayerParameter& param)
	  : BaseImageDataLayer<Dtype>(param) {}
	virtual ~FCNImageDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "FCNImageData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	inline int GetTestIterations(){
		return data_provider_.GetTestIterations();
	}
	inline int GetTotalSampleSize(){
		return data_provider_.GetPosSampleSize();
	}

	void ShowDataAndPredictedLabel(const string & output_path, const Blob<Dtype>& data,
			const Blob<Dtype>& label,const Blob<Dtype>& predicted, Dtype threshold);
 protected:
	ImageDataSourceBootstrapableProvider<Dtype> data_provider_;
	bool need_detection_box_;
	bool need_keypoint_;
	bool need_ignore_box_;

	int	GetScaleBaseId();
	virtual void BatchSetUp(Batch<Dtype>* batch) ;
	virtual void BatchFinalize(Batch<Dtype>* batch) ;
	virtual void ProcessImg(Batch<Dtype>* batch,int item_id) ;

	vector<bool> SetPointTransformIgnoreFlag(const std::pair<std::string, vector<Dtype> >& cur_sample,
			ImageDataSourceSampleType sample_type);

	virtual void SetUpParameter(const FCNImageDataParameter& fcn_image_data_param);
	virtual int SetUpChannelInfo( const int channel_base_offset = 0);

	vector<int> chosen_scale_id_;

	vector<cv::Mat> cv_img_;
	vector<cv::Mat> cv_img_depth_;
	vector<cv::Mat> cv_img_original_;
	vector<cv::Mat> cv_img_original_depth_;

};


/**
 * @brief Provides data to the Net from image files.By alan
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ShowImgPairLayer : public Layer<Dtype>
{
 public:
	explicit ShowImgPairLayer(const LayerParameter& param): Layer<Dtype>(param){}
	virtual ~ShowImgPairLayer();
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ShowImgPair"; }

	virtual inline int ExactNumTopBlobs() const { return 0; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};

	string save_folder_;
	int gt_blob_id_;
	Dtype mean_bgr_[3];
	int total_img_num_;

	int cur_count_;
	int cur_epoch_;
};


template <typename Dtype>
class BGR2GrayLayer : public Layer<Dtype>{
public:
	  explicit BGR2GrayLayer(const LayerParameter& param)
	      : Layer<Dtype>(param) {}
	  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	  virtual inline const char* type() const { return "BGR2Gray"; }
	  virtual inline int ExactNumBottomBlobs() const { return 1; }
	  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
	  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
	  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){};
	  Blob<Dtype> bgr_weight_;

};

class Point4D {
public:
	int n,c,y,x;
	Point4D ( int n_,int c_,int y_, int x_){
		n = n_;
		c = c_;
		y = y_;
		x = x_;
	}
};

/**
 * @brief  aka hard negative mining layer.It also offers capability to balance
 * 		   negative and positive samples. In current implementation, ignore flags
 * 		   for feature map is also supported.
 *
 *
 * @param bottom		   bottom[0] is the feature map
 * 		   				   you want to perform negative mining.
 * 		   				   bottom[1] is the label.
 * 		   				   The ratio of hard negative sample, random negative sample
 * 		  				   is controlled bu negative_slope_ and hard_ratio_;
 * 		  				   ignore_largest_n will ignore the n  largest mismatched sample .
 * 		  				   bottom[2] is the ignore flags. It should have the same structure as both
 * 		  				   bottom[0]  and bottom[1]..
 *
 * */
template <typename Dtype>
class LabelRelatedDropoutLayer : public Layer <Dtype> {
public:
	explicit LabelRelatedDropoutLayer(const LayerParameter & param)
			:Layer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "LabelRelatedDropout"; }
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

	static bool comparePointScore(const std::pair< Point4D ,Dtype>& c1,
	    const std::pair< Point4D ,Dtype>& c2);

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	void set_mask_for_positive_cpu(const vector<Blob<Dtype>*>& bottom);
	void set_mask_for_positive_gpu(const vector<Blob<Dtype>*>& bottom);

	void set_mask_from_labels_cpu(const vector<Blob<Dtype>*>& bottom);

	void get_all_pos_neg_instances(const vector<Blob<Dtype>*>& bottom,
			vector<int>& pos_count_in_channel );
	vector<int> get_permutation(int start, int end, bool random);
	void PrintMask();



	void set_mask_from_labels_cpu_parallel(const vector<Blob<Dtype>*>& bottom);
	void set_mask_from_labels_per_channel_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<int>& pos_count_in_channel,int channel_start_id, int interval);
	void get_all_pos_neg_instances_parallel(const vector<Blob<Dtype>*>& bottom,
				vector<int>& pos_count_in_channel );
	void get_all_pos_neg_instances_per_channel(const vector<Blob<Dtype>*>& bottom,
				vector<int>* pos_count_in_channel,int channel_start_id, int interval);

	int num_thread_;
	boost::shared_mutex mutex_;
//	vector<boost::thread * > get_all_pos_neg_instances_threads_;
//	vector<boost::thread * > set_mask_from_labels_threads_;


	Blob<int> mask_vec_; ///< the mask for backpro

	Dtype negative_ratio_;
	Dtype hard_ratio_;
	vector< vector <std::pair< Point4D ,Dtype> > > negative_points_;
	Dtype value_masked_; ///< the value of masked point in feature map is set to value_masked_
	int ignore_largest_n;
	int num_;
	int channels_;
	int height_;
	int width_;
	int margin_;
	bool pic_print_;
	string show_output_path_;
	int min_neg_nums_;
};


}  // namespace caffe

#endif  // CAFFE_FCN_DATA_LAYERS_HPP_

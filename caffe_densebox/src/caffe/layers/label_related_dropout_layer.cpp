// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layers/fcn_data_layers.hpp"
#include <boost/bind.hpp>
#include <boost/thread.hpp>
namespace caffe {

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  this->num_thread_ = this->layer_param_.label_related_dropout_param().num_thread();
  CHECK_GE(num_thread_,1);
  if(this->num_thread_ > 100){
	  LOG(INFO)<<"Warning, too many threads for LabelRelatedDropoutLayer layer.";
  }
  this->negative_ratio_ = (Dtype) this->layer_param_.label_related_dropout_param().negative_ratio();
  this->value_masked_ = (Dtype) this->layer_param_.label_related_dropout_param().value_masked();
  this->ignore_largest_n = this->layer_param_.label_related_dropout_param().ignore_largest_n();
  this->hard_ratio_ =   this->layer_param_.label_related_dropout_param().hard_ratio();
  CHECK(negative_ratio_ >= 0.);
  CHECK(negative_ratio_ < 1.);
  CHECK(this->hard_ratio_ >= 0);
  CHECK(this->hard_ratio_ <= 1);
  CHECK(ignore_largest_n >= 0);

  this->num_ = bottom[0]->num();
  this->channels_ = bottom[0]->channels();
  this->height_ = bottom[0]->height();
  this->width_ = bottom[0]->width();
  this->margin_ = this->layer_param_.label_related_dropout_param().margin();
  this->min_neg_nums_ = this->layer_param_.label_related_dropout_param().min_neg_nums();

  LOG(INFO)<<"num_thread: "<<num_thread_;

  this->negative_points_.clear();
  for(int i = 0; i< bottom[0]->channels(); i++)
  {
	  vector <std::pair< Point4D ,Dtype> > list_temp;
	  this->negative_points_.push_back(list_temp);

  }
  pic_print_ = ((getenv("LABEL_RELATED_DROPOUT_PRINT") != NULL) && (getenv("LABEL_RELATED_DROPOUT_PRINT")[0] == '1'));
  show_output_path_ = string("cache/LabelRelatedDropout");
  if (pic_print_ ) {
  		CreateDir(show_output_path_.c_str());
  }
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  // Set up the cache for mask
  this->mask_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());

  CHECK(bottom[0]->count() == bottom[1]->count());
  bottom[1]->Reshape(bottom[0]->shape());

  CHECK(bottom[0]->channels() == bottom[1]->channels());
  CHECK(bottom[0]->height() == bottom[1]->height());
  CHECK(bottom[0]->width() == bottom[1]->width());

  if( bottom.size() == 3)
  {
	  	CHECK(bottom[0]->channels() == bottom[2]->channels());
	    CHECK(bottom[0]->height() == bottom[2]->height());
	    CHECK(bottom[0]->width() == bottom[2]->width());
  }

}


template <typename Dtype>
bool LabelRelatedDropoutLayer<Dtype>::comparePointScore(const std::pair< Point4D,Dtype>& c1,
    const std::pair< Point4D,Dtype>& c2) {
  return c1.second >= c2.second;
}

template <typename Dtype>
vector<int> LabelRelatedDropoutLayer<Dtype>::get_permutation(int start, int end, bool random)
{
	vector<int> res;
	for(int i= start; i < end; ++i)
		res.push_back(i);
	if(random){
		boost::unique_lock<boost::shared_mutex> lock(this->mutex_);
		caffe::shuffle(res.begin(),res.end());
		lock.unlock();
	}
	return res;
}

template<typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::set_mask_for_positive_cpu(const vector<Blob<Dtype>*>& bottom){
	const Dtype * label_data = bottom[1]->cpu_data();
	int* mask_data = this->mask_vec_.mutable_cpu_data();
	for(int i=0; i < bottom[1]->count(); ++i){
		Dtype label_value = (label_data[i]);
		mask_data[i] = label_value == Dtype(0)? 0:1 ;
	}
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::get_all_pos_neg_instances(const vector<Blob<Dtype>*>& bottom,
		vector<int>& pos_count_in_channel  )
{
	pos_count_in_channel.resize(bottom[0]->channels(),0);

//	int pos_count = 0;
//	int all_neg = 0;
	const Dtype * label_data = bottom[1]->cpu_data();
	const Dtype * predicted_data = bottom[0]->cpu_data();
	Dtype * ignore_labels = bottom.size() > 2? bottom[2]->mutable_cpu_data() : NULL;

	int* mask_data = this->mask_vec_.mutable_cpu_data();


//		int cur_pos = 0;
	for(int c_id = 0 ; c_id < channels_; ++ c_id){
		for(int n_id = 0; n_id < num_; ++n_id){
			const Dtype* label_base_ptr = &(label_data[bottom[1]->offset(n_id,c_id)]);
			const Dtype* predicted_values_ptr = &(predicted_data[bottom[0]->offset(n_id, c_id)]);
			Dtype* ignore_label_ptr = NULL;

			if(ignore_labels != NULL)
				 ignore_label_ptr = &(ignore_labels[bottom[2]->offset(n_id, c_id)]);
			int* mask_ptr = mask_data + mask_vec_.offset(n_id,c_id);

			for(int cur_off = 0; cur_off < height_*width_; ++cur_off){
				// if counter an ignored instance
				if(ignore_label_ptr != NULL && static_cast<int>(ignore_label_ptr[cur_off]) != 0)
					continue;
				// if counter an positive instance
				if( (label_base_ptr[cur_off]) != Dtype(0)){
					mask_ptr[cur_off]= 1;
	//					++cur_pos;
					++pos_count_in_channel[c_id];
				}
				// for negative instances
				else{
					bool find_positive = false;
					for(int margin_row = margin_*-1;margin_row <= margin_; ++margin_row){
						for(int margin_col = margin_*-1; margin_col <= margin_; ++margin_col){
							int row_id = std::min(std::max(cur_off/width_+margin_row,0),height_-1);
							int col_id = std::min(std::max(cur_off%width_+margin_col,0),width_-1);
							if(label_base_ptr[row_id*width_+col_id] != Dtype(0)){
								find_positive = true;
								break;
							}
						}
						if(find_positive)
							break;
					}
					if(find_positive == false){
						negative_points_.at(c_id).push_back(std::make_pair(Point4D (n_id,c_id,
							cur_off/width_, cur_off%width_),predicted_values_ptr[cur_off]));
					}
				}
			}
		}
	}

//		if(cur_pos == 0)
//			all_neg+=1;
//		pos_count+=cur_pos;

}



template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::get_all_pos_neg_instances_per_channel(const vector<Blob<Dtype>*>& bottom,
		vector<int>* pos_count_in_channel,int channel_start_id, int interval){

	const Dtype * label_data = bottom[1]->cpu_data();
	const Dtype * predicted_data = bottom[0]->cpu_data();
	Dtype * ignore_labels = bottom.size() > 2? bottom[2]->mutable_cpu_data() : NULL;
	int* mask_data = this->mask_vec_.mutable_cpu_data();

	for(int c_id = channel_start_id ; c_id < channels_; c_id += this->num_thread_){
		for(int n_id = 0; n_id < num_; ++n_id){
			const Dtype* label_base_ptr = &(label_data[bottom[1]->offset(n_id,c_id)]);
			const Dtype* predicted_values_ptr = &(predicted_data[bottom[0]->offset(n_id, c_id)]);
			Dtype* ignore_label_ptr = NULL;
			if(ignore_labels != NULL)
				 ignore_label_ptr = &(ignore_labels[bottom[2]->offset(n_id, c_id)]);
			int* mask_ptr = mask_data + mask_vec_.offset(n_id,c_id);

			for(int cur_off = 0; cur_off < height_*width_; ++cur_off){
				// if counter an ignored instance
				if(ignore_label_ptr != NULL && static_cast<int>(ignore_label_ptr[cur_off]) != 0)
					continue;
				// if counter an positive instance
				if( (label_base_ptr[cur_off]) != Dtype(0)){
					mask_ptr[cur_off]= 1;
					++pos_count_in_channel->at(c_id);
//					LOG(INFO)<<"find pos at "<< cur_off <<" in c_id == "<<c_id;
				}
				// for negative instances
				else{
					bool find_positive = false;
					for(int margin_row = margin_*-1;margin_row <= margin_; ++margin_row){
						for(int margin_col = margin_*-1; margin_col <= margin_; ++margin_col){
							int row_id = std::min(std::max(cur_off/width_+margin_row,0),height_-1);
							int col_id = std::min(std::max(cur_off%width_+margin_col,0),width_-1);
							if(label_base_ptr[row_id*width_+col_id] != Dtype(0)){
								find_positive = true;
								break;
							}
						}
						if(find_positive)
							break;
					}
					if(find_positive == false){
						negative_points_.at(c_id).push_back(std::make_pair(Point4D (n_id,c_id,
							cur_off/width_, cur_off%width_),predicted_values_ptr[cur_off]));
					}
				}
			}
		}
	}

//	for(int i=0; i < pos_count_in_channel->size(); ++i){
//		LOG(INFO)<<"in get_all_pos_neg_instances_parallel, pos_count_in_channel["<<i<<"]:"<<pos_count_in_channel->at(i);
//	}
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::get_all_pos_neg_instances_parallel(const vector<Blob<Dtype>*>& bottom,
		vector<int>& pos_count_in_channel  ){
	pos_count_in_channel.resize(bottom[0]->channels(),0);
	vector<boost::thread * > get_all_pos_neg_instances_threads;
	for(int i=0; i < this->num_thread_; ++i){
		get_all_pos_neg_instances_threads.push_back(new boost::thread(boost::bind(
			&LabelRelatedDropoutLayer::get_all_pos_neg_instances_per_channel, this,
			bottom,&pos_count_in_channel,i, num_thread_)));
	}
	for(int i=0; i < this->num_thread_; ++i){
		try {
			get_all_pos_neg_instances_threads[i]->join();
			delete  get_all_pos_neg_instances_threads[i];
		}catch (...) {
			LOG(INFO)<<"joint thread failed in LabelRelatedDropoutLayer<Dtype>::get_all_pos_neg_instances_parallel.";
		}
	}
	get_all_pos_neg_instances_threads.clear();
//
//	for(int i=0; i < pos_count_in_channel.size(); ++i){
//			LOG(INFO)<<"at the end of  get_all_pos_neg_instances_parallel, pos_count_in_channel["<<i<<"]:"<<pos_count_in_channel[i];
//	}

}

template<typename Dtype>
void   LabelRelatedDropoutLayer<Dtype>::set_mask_from_labels_cpu(const vector<Blob<Dtype>*>& bottom)
{

//	if(Caffe::phase() == Caffe::TEST){
//		hard_ratio = 0;
//	}

	int* mask_data = this->mask_vec_.mutable_cpu_data();
	for(int i=0;i<channels_;++i){
		this->negative_points_.at(i).clear();
	}
	caffe_set(this->mask_vec_.count(),0,mask_data);
	vector<int> pos_count_channel(bottom[0]->channels(),0);
	get_all_pos_neg_instances(bottom,pos_count_channel);

	// assign mask
	for(int c=0; c< bottom[0]->channels();++c){
		// get negative count according to negative ratio
		int neg_count = std::max(0,int(pos_count_channel[c]/(1-this->negative_ratio_) - pos_count_channel[c]) );
		neg_count = std::max(neg_count,min_neg_nums_);
		neg_count = std::min(neg_count,(int)negative_points_.at(c).size()- ignore_largest_n);
		CHECK(neg_count <= negative_points_.at(c).size()- ignore_largest_n)<<"neg_count is larger than negative_points_.size()";
		std::stable_sort(negative_points_.at(c).begin(), negative_points_.at(c).end(),LabelRelatedDropoutLayer<Dtype>::comparePointScore);

		int hard_count = std::min(int(neg_count*hard_ratio_),neg_count);
		hard_count = std::max((hard_ratio_ >0  && hard_count > 1)? 1:0 ,hard_count);

		// sample the hard_count hard negative samples  from the top  neg_count hard negative candidates
		int hard_count_sampling_range = (std::min((unsigned int)(neg_count*1),(unsigned int)(negative_points_.at(c).size() - ignore_largest_n)));
		hard_count_sampling_range = std::max(0,hard_count_sampling_range);
		vector<int> hard_negative_random_ids = get_permutation(ignore_largest_n,ignore_largest_n + hard_count_sampling_range, true);
		for(int hard_neg_id = 0 ; hard_neg_id < hard_count; ++hard_neg_id){
			int id = hard_negative_random_ids[hard_neg_id];
			Point4D point =negative_points_.at(c)[id ].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_vec_.offset(point.n,point.c,point.y, point.x)]= 1;
		}

		// sample the remaining negative samples from all the negative candidates
		hard_negative_random_ids = get_permutation(ignore_largest_n, negative_points_.at(c).size(), true);
		for(int neg_id = 0; neg_id < neg_count-hard_count; neg_id++){
			int id = hard_negative_random_ids[neg_id];
			Point4D point =negative_points_.at(c)[id ].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_vec_.offset(point.n,point.c,point.y, point.x)]= 1;
		}
		if(this->pic_print_)
			LOG(INFO)<<"in channel "<<c<<" , neg_points size:" <<negative_points_.at(c).size()<<" neg_count: "<<neg_count <<"  pos_count "<<pos_count_channel[c]
						<<"  total size: "<<neg_count+pos_count_channel[c]  <<" hard_negative: "<<hard_count;
	}

}


template<typename Dtype>
void   LabelRelatedDropoutLayer<Dtype>:: set_mask_from_labels_per_channel_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<int>& pos_count_channel,int channel_start_id, int interval){
	int* mask_data = this->mask_vec_.mutable_cpu_data();

//	for(int i=0; i < pos_count_channel.size(); ++i){
//			LOG(INFO)<<"in set_mask_from_labels_per_channel_cpu, pos_count_channel["<<i<<"]:"<<pos_count_channel[i];
//	}
	// assign mask
	for(int c=channel_start_id; c< bottom[0]->channels();c += num_thread_){
		// get negative count according to negative ratio
		int neg_count = std::max(0,int(pos_count_channel[c]/(1-this->negative_ratio_) - pos_count_channel[c]) );
		neg_count = std::max(neg_count,min_neg_nums_);
		neg_count = std::min(neg_count,(int)negative_points_.at(c).size()- ignore_largest_n);
		CHECK(neg_count <= negative_points_.at(c).size()- ignore_largest_n)<<"neg_count is larger than negative_points_.size()";
		std::stable_sort(negative_points_.at(c).begin(), negative_points_.at(c).end(),LabelRelatedDropoutLayer<Dtype>::comparePointScore);

		int hard_count = std::min(int(neg_count*hard_ratio_),neg_count);
		hard_count = std::max((hard_ratio_ >0  && hard_count > 1)? 1:0 ,hard_count);

		// sample the hard_count hard negative samples  from the top  neg_count hard negative candidates
		int hard_count_sampling_range = (std::min((unsigned int)(neg_count*1),(unsigned int)(negative_points_.at(c).size() - ignore_largest_n)));
		hard_count_sampling_range = std::max(0,hard_count_sampling_range);
		vector<int> hard_negative_random_ids = get_permutation(ignore_largest_n,ignore_largest_n + hard_count_sampling_range, true);
		for(int hard_neg_id = 0 ; hard_neg_id < hard_count; ++hard_neg_id){
			int id = hard_negative_random_ids[hard_neg_id];
			Point4D point =negative_points_.at(c)[id ].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_vec_.offset(point.n,point.c,point.y, point.x)]= 1;
		}

		// sample the remaining negative samples from all the negative candidates
		hard_negative_random_ids = get_permutation(ignore_largest_n, negative_points_.at(c).size(), true);
		for(int neg_id = 0; neg_id < neg_count-hard_count; neg_id++){
			int id = hard_negative_random_ids[neg_id];
			Point4D point =negative_points_.at(c)[id ].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_vec_.offset(point.n,point.c,point.y, point.x)]= 1;
		}
		if(this->pic_print_)
			LOG(INFO)<<"in channel "<<c<<" , neg_points size:" <<negative_points_.at(c).size()<<" neg_count: "<<neg_count <<"  pos_count "<<pos_count_channel[c]
						<<"  total size: "<<neg_count+pos_count_channel[c]  <<" hard_negative: "<<hard_count;
	}

}

template<typename Dtype>
void   LabelRelatedDropoutLayer<Dtype>::set_mask_from_labels_cpu_parallel(const vector<Blob<Dtype>*>& bottom){
	int* mask_data = this->mask_vec_.mutable_cpu_data();
	for(int i=0;i<channels_;++i){
		this->negative_points_.at(i).clear();
	}
	caffe_set(this->mask_vec_.count(),0,mask_data);
	vector<int> pos_count_channel(bottom[0]->channels(),0);
	get_all_pos_neg_instances_parallel(bottom,pos_count_channel);

//	for(int i=0; i < pos_count_channel.size(); ++i){
//		LOG(INFO)<<"in set_mask_from_labels_cpu_parallel, after get_all_pos_neg_instances_parallel, pos_count_channel["<<i<<"]:"<<pos_count_channel[i];
//	}

	vector<boost::thread * > set_mask_from_labels_threads;
	for(int i=0; i < this->num_thread_; ++i){
		set_mask_from_labels_threads.push_back(new boost::thread(boost::bind(
			&LabelRelatedDropoutLayer::set_mask_from_labels_per_channel_cpu, this,
			bottom,pos_count_channel,i, num_thread_)));
	}
	for(int i=0; i < this->num_thread_; ++i){
		try {
			set_mask_from_labels_threads[i]->join();
			delete  set_mask_from_labels_threads[i];
		}catch (...) {
			LOG(INFO)<<"joint thread failed in LabelRelatedDropoutLayer<Dtype>::get_all_pos_neg_instances_parallel.";
		}
	}
	set_mask_from_labels_threads.clear();
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();

	if(this->negative_ratio_ <= 0)
		set_mask_for_positive_cpu(bottom);
	else{
		this->set_mask_from_labels_cpu_parallel(bottom);
	}
	if(this->pic_print_)
		PrintMask();
	const int* mask = this->mask_vec_.cpu_data();
	for (int i = 0; i < count; ++i) {
	  top_data[i] = bottom_data[i] * mask[i] + (mask[i] ==0)*value_masked_;
	}
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  const Dtype* top_diff = top[0]->cpu_diff();
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      const  int* mask = this->mask_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] ;
      }
  }
}

template <typename Dtype>
void LabelRelatedDropoutLayer<Dtype>::PrintMask()
{
	const LayerParameter& layer_param = this->layer_param();
	string layer_name = layer_param.name();
	const  int* mask_value = this->mask_vec_.cpu_data();
	char path[512];
	vector<int> params_jpg;
	params_jpg.push_back(CV_IMWRITE_JPEG_QUALITY);
	params_jpg.push_back(100);
	for(int n = 0 ; n <  mask_vec_.num(); ++n){
		for(int c = 0; c <  mask_vec_.channels(); ++c){
			cv::Mat mask_img = cv::Mat(mask_vec_.height(), mask_vec_.width(), CV_8UC1);
			for(int h = 0; h < mask_vec_.height();++h){
				for(int w = 0 ; w < mask_vec_.width(); ++w){
					mask_img.at<uchar>(h,w) = 255 * MIN(1, MAX(0, mask_value[mask_vec_.offset(n,c,h,w)]));
				}
			}
			sprintf(path,"%s/%s_mask_num_%03d_channel_%03d.jpg", this->show_output_path_.c_str(), layer_name.c_str(),n,c);
			imwrite(path, mask_img, params_jpg);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(LabelRelatedDropoutLayer);
#endif

INSTANTIATE_CLASS(LabelRelatedDropoutLayer);
REGISTER_LAYER_CLASS(LabelRelatedDropout);

}  // namespace caffe

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_others.hpp"

namespace caffe {

template <typename Dtype>
ShowImgPairLayer<Dtype>::~ShowImgPairLayer(){

}

template <typename Dtype>
void ShowImgPairLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
	ShowImgPairParam show_img_pair_param = this->layer_param_.show_img_pair_param();
	this->save_folder_ = show_img_pair_param.save_folder();
	this->gt_blob_id_ = show_img_pair_param.gt_blob_id();
	this->mean_bgr_[0] = show_img_pair_param.mean_b();
	this->mean_bgr_[1] = show_img_pair_param.mean_g();
	this->mean_bgr_[2] = show_img_pair_param.mean_r();
	this->total_img_num_ = show_img_pair_param.total_img_num();
	this->cur_count_ = 0;
	this->cur_epoch_ = 0;

	CreateDir(save_folder_.c_str());
	CHECK_LT(this->gt_blob_id_, bottom.size());
	CHECK_EQ(bottom[0]->num(), bottom[1]->num());
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
}

template <typename Dtype>
void ShowImgPairLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


}

template <typename Dtype>
void ShowImgPairLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	char output_path[512];
	for(int n_id = 0; n_id < bottom[0]->num(); ++n_id){
//		LOG(INFO)<<"cur_count_ "<<cur_count_<<"  start";
		for(int i=0; i < bottom.size(); ++i){
			if(i == this->gt_blob_id_){
				sprintf(output_path, "%s/%05d-%05d_gt.jpg",this->save_folder_.c_str(),this->cur_count_,this->cur_epoch_);
			}else{
				sprintf(output_path, "%s/%05d-%05d.jpg",this->save_folder_.c_str(),this->cur_count_,this->cur_epoch_);
			}
			cv::Mat temp_mat = BlobImgDataToCVMat(*(bottom[i]),n_id, mean_bgr_[0],mean_bgr_[1],mean_bgr_[2]);
			cv::imwrite(output_path,temp_mat);
		}
//		LOG(INFO)<<"cur_count_ "<<cur_count_<<"  finished";
		cur_count_++;
		if(cur_count_ == this->total_img_num_){
			cur_epoch_++;
			cur_count_ = 0;
		}
	}

}

template <typename Dtype>
void ShowImgPairLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	Forward_cpu( bottom, top);
}

#ifdef CPU_ONLY
STUB_GPU(ShowImgPairLayer);
#endif

INSTANTIATE_CLASS(ShowImgPairLayer);
REGISTER_LAYER_CLASS(ShowImgPair);

}  // namespace caffe

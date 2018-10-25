#include <string>
#include <vector>
#include <stdint.h>

#include "caffe/layers/fcn_data_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include "boost/threadpool.hpp"
namespace caffe {


template <typename Dtype>
BaseImageDataLayer<Dtype>::~BaseImageDataLayer<Dtype>() {
	this->StopInternalThread();
}

template <typename Dtype>
void BaseImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top)
{
	const FCNImageDataParameter& fcn_image_data_param =
			this->layer_param_.fcn_image_data_param();
	CHECK(fcn_image_data_param.has_fcn_image_data_common_param())<<"fcn_image_data_comon_param is necessary";
	CHECK(fcn_image_data_param.has_fcn_image_data_source_param())<<"fcn_img_data_source_param is necessary";

	const  FCNImageDataCommonParameter & fcn_img_data_common_param =
			fcn_image_data_param.fcn_image_data_common_param();
	const FCNImageDataSourceParameter & fcn_img_data_source_param =
			fcn_image_data_param.fcn_image_data_source_param();
	const FCNImageDataReaderParameter & fcn_img_data_reader_param =
			fcn_image_data_param.fcn_image_data_reader_param();

	this->single_thread_ = fcn_img_data_common_param.single_thread();
	this->batch_size_ = fcn_img_data_source_param.batch_size();
	this->thread_num_ = 20;
	this->is_data_in_gpu_ = fcn_img_data_reader_param.use_gpu() && fcn_img_data_reader_param.is_img_pair();

	need_prefetch_bbox_ = false;
	if(top.size() > 2){
		need_prefetch_bbox_ = true;
		top[2]->Reshape(1,1,1,6);
	}

//	  LOG(INFO)<<"top.size : "<<top.size() <<"(top.size() > 2) ? "<<(top.size() > 2) <<
//			  " need_prefetch_bbox_: "<<need_prefetch_bbox_;
//	CHECK(0);

//	img_process_threads_.clear();
	thread_pool_.size_controller().resize(thread_num_);

	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    	this->prefetch_[i].bbox_.Reshape(1,1,1,6);
      this->prefetch_[i].bbox_.mutable_cpu_data();
    }
}



template <typename Dtype>
void BaseImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
	this->BatchSetUp(batch);
	for(int i=0; i < this->batch_size_; ++i)
	{
		if(single_thread_)
		{
			ProcessImg(batch,i);
		}
		else
		{
			try {
				thread_pool_.schedule(boost::bind(&BaseImageDataLayer::ProcessImg, this,batch, i));
			}catch (...) {
				LOG(INFO)<<"Create thread failed.";
			}
		}
	}
	if(! single_thread_)
	{
		try {
			thread_pool_.wait();
		}catch (...) {
			LOG(INFO)<<"joint thread failed.";
		}
	}
	this->BatchFinalize(batch);
}


template <typename Dtype>
void BaseImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
	Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
	top[0]->ReshapeLike(batch->data_);

	caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
	             top[0]->mutable_cpu_data());

  if(batch->label_.channels() > 0 && top.size()>1){
  	top[1]->ReshapeLike(batch->label_);
		caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
				top[1]->mutable_cpu_data());
  }

  DLOG(INFO) << "Prefetch copied";

  if (this->need_prefetch_bbox_) {
  	top[1]->ReshapeLike(batch->bbox_);
		caffe_copy(batch->bbox_.count(), batch->bbox_.cpu_data(),
				top[2]->mutable_cpu_data());
  }

  last_batch_samples_.clear();
  vector<std::pair<std::string, vector<Dtype> > > & buff_batch_samples_ =
  		batch->buff_batch_samples_;
  for(int i=0; i < buff_batch_samples_.size(); ++i){
		last_batch_samples_.push_back(buff_batch_samples_[i]);
  }

  this->prefetch_free_.push(batch);
}


template <typename Dtype>
void BaseImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


	Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
	top[0]->ReshapeLike(batch->data_);

	if(is_data_in_gpu_){
		caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
				top[0]->mutable_gpu_data());
	}else{
		caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
				top[0]->mutable_gpu_data());
	}


	if(batch->label_.channels() > 0 && top.size()>1){
		top[1]->ReshapeLike(batch->label_);
		caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
				top[1]->mutable_gpu_data());
	}

  DLOG(INFO) << "Prefetch copied";

  if (this->need_prefetch_bbox_) {
  	top[1]->ReshapeLike(batch->bbox_);
		caffe_copy(batch->bbox_.count(), batch->bbox_.cpu_data(),
				top[2]->mutable_gpu_data());
  }

  last_batch_samples_.clear();
  vector<std::pair<std::string, vector<Dtype> > > & buff_batch_samples_ =
  		batch->buff_batch_samples_;
  for(int i=0; i < buff_batch_samples_.size(); ++i){
		last_batch_samples_.push_back(buff_batch_samples_[i]);
  }

  this->prefetch_free_.push(batch);


}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BaseImageDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseImageDataLayer);
}  // namespace caffe

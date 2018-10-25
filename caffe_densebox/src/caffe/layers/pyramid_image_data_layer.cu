#include <string>
#include <vector>
#include <algorithm>
#include "caffe/layers/fcn_data_layers.hpp"
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




/**
 *
 */
template <typename Dtype>
void PyramidImageDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top){

	Timer total, prefetch;
	total.Start();
	prefetch.Start();

	if(forward_iter_id_ == 0 ){
		StartProcessOneImg();
	}
	Dtype prefetch_time = prefetch.MicroSeconds() ;
	Blob<Dtype>& img_blob_ = *(img_blob_list_[used_buffered_block_id_]);
	Blob<Dtype>& buff_block = *(buffered_block_[used_buffered_block_id_]);

	int start_id = forward_iter_id_ * max_block_num_;
	int end_id = MIN(forward_iter_id_ * max_block_num_ +max_block_num_ ,buff_block.num());

	top[0]->Reshape(end_id - start_id,img_blob_.channels(),
			buff_block.height(), buff_block.width());
	int block_length = img_blob_.channels() *buff_block.height()*buff_block.width();
	for(int i= start_id; i < end_id; ++i){
		caffe::caffe_copy(block_length,buff_block.gpu_data()+buff_block.offset(i,0),
				top[0]->mutable_gpu_data()+top[0]->offset(i-start_id,0));
	}
	caffe_set(top[1]->count(),Dtype(0),top[1]->mutable_cpu_data());
	SerializeToBlob(*(top[1]), 0);
	if (this->pic_print_)
		this->ShowImg(top);
	forward_iter_id_ =  (forward_iter_id_+1) % forward_times_for_cur_sample_;
	if(show_time_){
		LOG(INFO)<<"Time for PyramidImageDataLayer::Forward_gpu:  "<<total.MicroSeconds()/1000.0<<
				" milliseconds, in which prefetch cost "<< prefetch_time/1000;
	}
}





INSTANTIATE_LAYER_GPU_FUNCS(PyramidImageDataLayer);


}  // namespace caffe

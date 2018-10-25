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
#include <stdint.h>
namespace caffe {


template <typename Dtype>
__global__ void mat_to_blog_kernel(const int data_dim,const uint8_t* img_data,
		Dtype* dst_data, const int channel, const int height, const int width,
		const Dtype mean_b, const Dtype mean_g, const Dtype mean_r){
	Dtype mean_bgr[3];
	mean_bgr[0] = mean_b;
	mean_bgr[1] = mean_g;
	mean_bgr[2] = mean_r;
	CUDA_KERNEL_LOOP(index, data_dim){
		int c_id = index % channel;
		int h_id = (index/channel)/width;
		int w_id = (index/channel)%width;
		dst_data[(c_id * height + h_id) * width + w_id ] = (static_cast<Dtype>(img_data[index]) - mean_bgr[c_id]);
	}
}


template <typename Dtype>
void PyramidImageOnlineDataLayer<Dtype>::LoadOneImgToInternalBlob_gpu(const cv::Mat& img){
	int buff_block_id = (this->used_buffered_block_id_+1)%(this->buffered_block_.size());
	Blob<Dtype>& img_blob_ = *(this->img_blob_list_[buff_block_id]);
	img_blob_.Reshape(1,img.channels(),img.rows,img.cols);
	uint8_blob_.Reshape(1,img.channels(),img.rows,img.cols);
	const int count = img_blob_.count();
	caffe::caffe_copy(count,img.data,uint8_blob_.mutable_gpu_data());
	mat_to_blog_kernel<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, uint8_blob_.gpu_data(), img_blob_.mutable_gpu_data(),img.channels(),img.rows,img.cols,
			this->mean_bgr_[0],this->mean_bgr_[1],this->mean_bgr_[2]);

}





template void PyramidImageOnlineDataLayer<float>::LoadOneImgToInternalBlob_gpu(const cv::Mat& img);
template void PyramidImageOnlineDataLayer<double>::LoadOneImgToInternalBlob_gpu(const cv::Mat& img);



}  // namespace caffe

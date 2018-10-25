/*
 * caffe_wrapper.hpp

 *      Author: Alan_Huang
 */

#ifndef CAFFE_WRAPPER_COMMON_HPP_
#define CAFFE_WRAPPER_COMMON_HPP_

#include <iostream>
#include <vector>

namespace caffe{

struct ROIWithScale{
	float l,t,r,b,scale;
	ROIWithScale(){
		l=t=r=b=scale=0;
	}
	ROIWithScale(float l_, float t_, float r_, float b_, float scale_){
		l=l_;t=t_;r=r_;b=b_; scale = scale_;
	}
	friend std::ostream& operator << (std::ostream & stream,const ROIWithScale & rect){
		stream<< "("<<rect.l<<","<<rect.t<<","<<rect.r<<","<< rect.b<<","<<rect.scale<<")";
		return stream;
	}
};


template <typename Dtype>
struct BBox{
	BBox(){
		id = center_h = center_w = score = x1 = x2 = y1 = y2 = 0;
	}
	Dtype score,x1,y1,x2,y2, center_h, center_w,id;
	std::vector<Dtype> data;
	static bool greater(const BBox<Dtype>& a, const BBox<Dtype>& b){
		return a.score > b.score;
	}
};


}




#endif /* CAFFE_WRAPPER_COMMON_HPP_ */

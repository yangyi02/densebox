/*
 * caffe_wrapper.hpp
 *
 *  Created on: 2016年2月15日
 *      Author: Alan_Huang
 */

#ifndef CAFFE_WRAPPER_HPP_
#define CAFFE_WRAPPER_HPP_

#include "caffe/caffe_wrapper_common.hpp"
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
namespace caffe{


class CaffeDenseBoxDetector{
public:
	explicit CaffeDenseBoxDetector(const std::string proto_name, const std::string model_name,
			const bool use_cuda = true);
	explicit CaffeDenseBoxDetector(const std::string proto_name, const bool use_cuda = true);

	~CaffeDenseBoxDetector();

	void CopyFromModel(const std::string model_name);

	int ClassNum();
	void SetCudaFlag(bool flag);

	bool LoadImgToBuffer(cv::Mat & src_img);
	bool SetRoiWithScale(const std::vector<ROIWithScale>& roi_scale);

	void PredictOneImg();

	std::vector< BBox<float> >& GetBBoxResults(int class_id);
	std::vector<std::vector< BBox<float> > >& GetBBoxResults();

private:
	CaffeDenseBoxDetector(const CaffeDenseBoxDetector&);
	CaffeDenseBoxDetector& operator=(const CaffeDenseBoxDetector&);

  void* net_;

};


}



#endif /* CAFFE_WRAPPER_HPP_ */

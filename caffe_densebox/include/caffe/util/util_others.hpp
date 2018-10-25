#ifndef CAFFE_UTIL_UTIL_OTHER_H_
#define CAFFE_UTIL_UTIL_OTHER_H_

#include <string>
#include <map>
#include <utility>
#include <vector>
#include "caffe/util/buffered_reader.hpp"
#include "caffe/caffe_wrapper_common.hpp"
using std::string;
using std::vector;
using std::pair;


namespace caffe {

//template <typename Dtype>
//struct BBox{
//	BBox(){
//		id = center_h = center_w = score = x1 = x2 = y1 = y2 = 0;
//	}
//	Dtype score,x1,y1,x2,y2, center_h, center_w,id;
//	vector<Dtype> data;
//	static bool greater(const BBox<Dtype>& a, const BBox<Dtype>& b){
//		return a.score > b.score;
//	}
//};


template <typename Dtype>
struct ScorePoint{
	ScorePoint(){
		score = x = y = accumulated_x= accumulated_y= accumulated_weight = 0;
		weight = 1;
	}
	ScorePoint(Dtype s,Dtype xx,Dtype yy){
		score = s, x = xx, y = yy;
		accumulated_x= accumulated_y= weight=accumulated_weight = 0;
	}
	Dtype score,x,y;
	int weight;
	Dtype accumulated_x,accumulated_y,accumulated_weight ;
	static bool greater(const ScorePoint<Dtype>& a, const ScorePoint<Dtype>& b){
		return a.score > b.score;
	}
	Dtype dist(ScorePoint<Dtype>& other){
		return sqrt((this->x - other.x)*(this->x - other.x) + (this->y - other.y)*(this->y - other.y));
	}
};

//INSTANTIATE_STRUCT(BBox);
/*
 * Return the bboxes(each is 4 dims, x1, y1, x2, y2) of the given coords
 * If no bbox found(all the annotations are -1), then return four -1
 */
void GetBBoxes(const vector<float>& coords, const int key_points_count, vector<vector<float> >& bboxes);

void GetBBoxStandardScale(const vector<float>& coords, const int key_points_count,
		const int standard_bbox_diagonal_len, vector<float>& standard_scale);

void GetAllBBoxStandardScale(const vector<std::pair<std::string, vector<float> > >& samples,
		const int key_points_count, const int standard_bbox_diagonal_len,
		vector<vector<float> >& bboxes_standard_scale);

template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
    const pair<Dtype, vector<float> >& c2);

template <typename Dtype>
bool compareCandidate_v2(const vector<Dtype>  & c1,
    const  vector<Dtype>  & c2);

/* @brief     Designed by Zhujin. Non-maximum suppression. return a mask which elements are selected
*   		   overlap   Overlap threshold for suppression
*             For a selected box Bi, all boxes Bj that are covered by
*             more than overlap are suppressed. Note that 'covered' is
*             is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over
*             union measure.
* 			   if addscore == true, then the scores of all the overlap bboxes will be added
*/
template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore = false);

//@brief       Non-maximum suppression. return a mask which elements are selected
//   		   overlap   Overlap threshold for suppression
//             For a selected box Bi, all boxes Bj that are covered by
//             more than overlap are suppressed. Note that 'covered' is
//             is |Bi \cap Bj| / min(|Bj|,|Bi|), n
// 			  if addscore == true, then the scores of all the overlap bboxes will be added

template <typename Dtype>
const vector<bool> nms(vector< vector<Dtype> >& candidates,
    const Dtype overlap, const int top_N, const bool addScore = false);

template <typename Dtype>
const vector<bool> nms(vector< BBox<Dtype> >& candidates,
    const Dtype overlap, const int top_N, const bool addScore = false);

template <typename Dtype>
const vector<bool> nms(vector< ScorePoint<Dtype> >& candidates,
    const Dtype dist_threshold, const int top_N, const bool addScore = false,const Dtype accumulate_dist = 1);


// @brief      bbox voting. Return a vector showing which element is selected
//             For a selected box Bi, all boxes Bj that are covered by
//             more than overlap are suppressed. Note that 'covered' is
//             is |Bi \cap Bj| / (|Bj|+|Bi| - Intersection(|Bj|,|Bi|)),
//   		   the bbox coordinates are averaged as well if any two bboxes
//			   are close enough( overlap ratio < threshold).

template <typename Dtype>
const vector<bool> bbox_voting(vector< vector<Dtype> >& candidates,
    const Dtype overlap);



template <typename Dtype>
Dtype GetArea(const vector<Dtype>& bbox);


template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2);

// intersection over union
template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2);


enum OverlapType{ OVERLAP_UNION,OVERLAP_BOX1, OVERLAP_BOX2 };
template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22,
		const OverlapType overlap_type);


// |bbox1 \cap bbox2| / |bbox2|
template <typename Dtype>
Dtype GetNMSOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2);

void GetPredictedWithGT_FDDB(const string gt_file, const string pred_file,
		vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		int &n_positive, bool showing = false, string img_folder = "", string output_folder = "",float ratio = 0.5);

float GetPRPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive,vector<float>& precision,vector<float> &recall);

float GetTPFPPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive,vector<float>& precision,vector<float> &recall);


cv::Scalar GetColorById(int id);
void ShowClassColor(vector<string> class_names, string& out_name);
/**
 * @brief  This function return a vector showing whether the candidate is correct according
 * 		   to overlap ratio.
 */
vector<bool> GetPredictedResult(const vector< std::pair<int, vector<float> > > &gt_instances,
		const vector< std::pair<float, vector<float> > > &pred_instances, float ratio = 0.5);
template <typename Dtype>
bool ShowBBoxOnImage(const string img_path, const vector< BBox<Dtype> >& bboxes,
		const Dtype threshold,const string out_path,const cv::Scalar color = cv::Scalar(255,0,0) ,
		const int thickness = 1,BufferedColorJPGReader<Dtype>* p_buffered_jpg_reader = NULL);

template <typename Dtype>
bool ShowMultiClassBBoxOnImage(const string img_path, const vector< vector< BBox<Dtype> > >& multiclass_bboxes,
		const vector<Dtype> multiclass_threshold,const string out_path,  int thickness = 1,
		BufferedColorJPGReader<Dtype>* p_buffered_jpg_reader = NULL );

template <typename Dtype>
void ShowBBoxOnMat(cv::Mat& img,const vector< BBox<Dtype> >& bboxes,const Dtype threshold,
		const cv::Scalar color = cv::Scalar(255,0,0) , const int thickness = 1);

template <typename Dtype>
void PushBBoxTo(std::ofstream & out_result_file,const vector< BBox<Dtype> >& bboxes);

std::vector<std::string> std_split( const std::string&  stri,const std::string& reg);



template <typename Dtype>
void DrawScorePointsOnMat(cv::Mat &mat,const vector<ScorePoint<Dtype> >& points,const cv::Scalar &color,
		const int r,const vector<int>& line_point_pairs, int thickness = 1);



}  // namespace caffe

#endif   // CAFFE_UTIL_UTIL_OTHER_H_

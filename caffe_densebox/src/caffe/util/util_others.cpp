#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

#ifndef ELLISION
#define ELLISION  1e-9
#endif
void GetBBoxes(const vector<float>& coords, const int key_points_count,
		vector<vector<float> >& bboxes) {

	CHECK(coords.size() % (key_points_count * 2) == 0)
			<< "The number of key points is wrong.";

	bboxes = vector<vector<float> >(coords.size() / (key_points_count * 2), vector<float>(4, -1));
	for (int i = 0; i < bboxes.size(); ++i) {
		float min_x = -1, max_x = -1;
		float min_y = -1, max_y = -1;
		for (int j = 0; j < key_points_count; ++j) {
			int idx = i * key_points_count + j;
			idx *= 2;
			if (std::abs(coords[idx] - (-1)) < ELLISION
					|| std::abs(coords[idx + 1] - (-1)) < ELLISION) continue;

			if (std::abs(min_x - (-1)) < ELLISION) {
				min_x = coords[idx];
				max_x = coords[idx];

				min_y = coords[idx + 1];
				max_y = coords[idx + 1];
			} else {
				min_x = MIN(min_x, coords[idx]);
				max_x = MAX(max_x, coords[idx]);

				min_y = MIN(min_y, coords[idx + 1]);
				max_y = MAX(max_y, coords[idx + 1]);
			}
		}
		bboxes[i][0] = min_x;
		bboxes[i][1] = min_y;

		bboxes[i][2] = max_x;
		bboxes[i][3] = max_y;
	}
}

void GetBBoxStandardScale(const vector<float>& coords, const int key_points_count,
		const int standard_bbox_diagonal_len, vector<float>& standard_scale) {

	standard_scale.clear();

	vector<vector<float> > bboxes;
	GetBBoxes(coords, key_points_count, bboxes);
	for (int j = 0; j < bboxes.size(); ++j) {
		if (std::abs(bboxes[j][0] - (-1)) < ELLISION) {
			standard_scale.push_back(1);
		} else {
			float w = bboxes[j][0] - bboxes[j][2];
			float h = bboxes[j][1] - bboxes[j][3];
			standard_scale.push_back(standard_bbox_diagonal_len / std::sqrt(w * w + h * h));
		}
	}
}

void GetAllBBoxStandardScale(const vector<std::pair<std::string, vector<float> > >& samples,
		const int key_points_count, const int standard_bbox_diagonal_len,
		vector<vector<float> >& bboxes_standard_scale) {

	bboxes_standard_scale = vector<vector<float> >(samples.size(), vector<float>());
	for (int i = 0; i < samples.size(); ++i) {
		GetBBoxStandardScale(samples[i].second, key_points_count, standard_bbox_diagonal_len,
				bboxes_standard_scale[i]);
	}
}


template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
    const pair<Dtype, vector<float> >& c2) {
  return c1.first >= c2.first;
}

template bool compareCandidate<float>(const pair<float, vector<float> >& c1,
    const pair<float, vector<float> >& c2);
template bool compareCandidate<double>(const pair<double, vector<float> >& c1,
    const pair<double, vector<float> >& c2);

template <typename Dtype>
bool compareCandidate_v2(const vector<Dtype>  & c1,
    const  vector<Dtype>  & c2) {
  return c1[0] >= c2[0];
}

template bool compareCandidate_v2(const vector<float>  & c1,
    const  vector<float>  & c2);
template bool compareCandidate_v2(const vector<double>  & c1,
    const  vector<double>  & c2);


template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore) {
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate<Dtype>);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].second[2] - candidates[i].second[0] + 1)
				* (candidates[i].second[3] - candidates[i].second[1] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = MAX(candidates[i].second[0], candidates[j].second[0]);
      float yy1 = MAX(candidates[i].second[1], candidates[j].second[1]);
      float xx2 = MIN(candidates[i].second[2], candidates[j].second[2]);
      float yy2 = MIN(candidates[i].second[3], candidates[j].second[3]);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
        float o = w * h / areas[j];
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].first += candidates[j].first;
          }
        }
      }
    }
  }

  return mask;
}

template const vector<bool> nms<float> (vector<pair<float, vector<float> > >& candidates,
    const float overlap, const int top_N, const bool addScore = false);
template const vector<bool> nms<double> (vector<pair<double, vector<float> > >& candidates,
		const float overlap, const int top_N, const bool addScore = false);


template <typename Dtype>
const vector<bool> bbox_voting(vector< vector<Dtype> >& candidates,
    const Dtype overlap){

	  vector<bool> mask(candidates.size(), false);
	  vector<int> voted_weight(candidates.size(), 1);

	  if (mask.size() == 0) return mask;
	  //LOG(INFO)<<"overlap: "<<overlap;
	  vector<bool> skip(candidates.size(), false);
	  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate_v2<Dtype>);

	  vector<Dtype> areas(candidates.size(), 0);
	  for (int i = 0; i < candidates.size(); ++i) {
	  	areas[i] = (candidates[i][3] - candidates[i][1] + 1)
					* (candidates[i][4] - candidates[i][2] + 1);
	  }

	  for (int count = 0, i = 0;   i < mask.size(); ++i) {
	    if (skip[i]) continue;

	    mask[i] = true;
	    ++count;

	    // suppress the significantly covered bbox
	    for (int j = i + 1; j < mask.size(); ++j) {
	      if (skip[j]) continue;

	      // get intersections
	      Dtype xx1 = MAX(candidates[i][1], candidates[j][1]);
	      Dtype yy1 = MAX(candidates[i][2], candidates[j][2]);
	      Dtype xx2 = MIN(candidates[i][3], candidates[j][3]);
	      Dtype yy2 = MIN(candidates[i][4], candidates[j][4]);
	      Dtype w = xx2 - xx1 + 1;
	      Dtype h = yy2 - yy1 + 1;
	      //LOG(INFO)<<"xx1:"<<xx1<<"  yy1:"<<yy1<<"  xx2:"<<xx2<<"  yy2:"<<yy2;
	      if (w > 0 && h > 0) {
	        // compute overlap
	    	Dtype o = w * h / (areas[j]+areas[i] - w * h);
	       // LOG(INFO)<<o;
	        if (o > overlap) {
	        	skip[j] = true;
	          	candidates[i][0] += candidates[j][0];
	          	voted_weight[i]+= 1;
	          	candidates[i][1] = (candidates[j][1] + candidates[i][1] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][2] = (candidates[j][2] + candidates[i][2] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][3] = (candidates[j][3] + candidates[i][3] * (voted_weight[i] - 1))/voted_weight[i];
	          	candidates[i][4] = (candidates[j][4] + candidates[i][4] * (voted_weight[i] - 1))/voted_weight[i];
	        }
	      }

	    }

	  }

	  return mask;
}

template const vector<bool> bbox_voting(vector< vector<float> >& candidates,
    const float overlap);
template const vector<bool> bbox_voting(vector< vector<double> >& candidates,
    const double overlap);


template <typename Dtype>
const vector<bool> nms(vector < vector<Dtype>   >& candidates,
    const Dtype overlap, const int top_N, const bool addScore) {

  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;
  //LOG(INFO)<<"overlap: "<<overlap;
  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate_v2<Dtype>);

  vector<Dtype> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i][3] - candidates[i][1] + 1)
				* (candidates[i][4] - candidates[i][2] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      Dtype xx1 = MAX(candidates[i][1], candidates[j][1]);
      Dtype yy1 = MAX(candidates[i][2], candidates[j][2]);
      Dtype xx2 = MIN(candidates[i][3], candidates[j][3]);
      Dtype yy2 = MIN(candidates[i][4], candidates[j][4]);
      Dtype w = xx2 - xx1 + 1;
      Dtype h = yy2 - yy1 + 1;
      //LOG(INFO)<<"xx1:"<<xx1<<"  yy1:"<<yy1<<"  xx2:"<<xx2<<"  yy2:"<<yy2;
      if (w > 0 && h > 0) {
        // compute overlap
    	Dtype o = w * h / std::min(areas[j],areas[i]);
       // LOG(INFO)<<o;
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i][0] += candidates[j][0];
          }
        }
      }
    }
  }
  return mask;
}

template const vector<bool> nms (vector < vector<float>   >& candidates,
    const float overlap, const int top_N, const bool addScore);

template const vector<bool> nms (vector < vector<double>   >& candidates,
    const double overlap, const int top_N, const bool addScore);


template <typename Dtype>
const vector<bool> nms(vector< BBox<Dtype> >& candidates,
    const Dtype overlap, const int top_N, const bool addScore) {
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].x2 - candidates[i].x1 + 1)
				* (candidates[i].y2- candidates[i].y1 + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = MAX(candidates[i].x1, candidates[j].x1);
      float yy1 = MAX(candidates[i].y1, candidates[j].y1);
      float xx2 = MIN(candidates[i].x2, candidates[j].x2);
      float yy2 = MIN(candidates[i].y2, candidates[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
        //float o = w * h / areas[j];
    	float o = w * h / std::min(areas[j],areas[i]);
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].score += candidates[j].score;
          }
        }
      }
    }
  }

  return mask;
}
template const vector<bool> nms  (vector< BBox<float> >&  candidates,
    const float overlap, const int top_N, const bool addScore );
template const vector<bool> nms  (vector< BBox<double> >&  candidates,
		const double overlap, const int top_N, const bool addScore );


template <typename Dtype>
const vector<bool> nms(vector< ScorePoint<Dtype> >& candidates,
		const Dtype dist_threshold, const int top_N, const bool addScore,
		const Dtype accumulate_dist){

	vector<bool> mask(candidates.size(), false);

	if (mask.size() == 0) return mask;

	vector<bool> skip(candidates.size(), false);
	std::stable_sort(candidates.begin(), candidates.end(), ScorePoint<Dtype>::greater);

	for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
		if (skip[i]) continue;

		mask[i] = true;
		++count;
		// suppress the significantly close point
		for (int j = i + 1; j < mask.size(); ++j) {
			if (skip[j]) continue;
			Dtype dist = candidates[i].dist(candidates[j]);
			if (dist < dist_threshold) {
			  skip[j] = true;
			  if (addScore) {
			  	candidates[i].score += candidates[j].score;
				if(dist < accumulate_dist){
					candidates[i].accumulated_x += candidates[j].x;
					candidates[i].accumulated_y += candidates[j].y;
					candidates[i].accumulated_weight ++;
				}
					candidates[i].weight ++;
			  }
			}
		}
	}

	return mask;
}

template const vector<bool> nms(vector< ScorePoint<float> >& candidates,
	    const float dist_threshold, const int top_N, const bool addScore,const float accumulate_dist  );
template const vector<bool> nms(vector< ScorePoint<double> >& candidates,
	    const double dist_threshold, const int top_N, const bool addScore,const double accumulate_dist  );


template <typename Dtype>
Dtype  GetArea(const vector<Dtype>& bbox) {
	Dtype w = bbox[2] - bbox[0] + 1;
	Dtype h = bbox[3] - bbox[1] + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	return w * h;
}

template float GetArea(const vector<float>& bbox);
template double GetArea(const vector<double>& bbox);

template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2)
{
	Dtype w = x2- x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	return w * h;
}

template float GetArea(const float x1, const float y1, const float x2, const float y2);
template double GetArea(const double x1, const double y1, const double x2, const double y2);


template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2) {
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(bbox1);
	Dtype area2 = GetArea(bbox2);
	Dtype u = area1 + area2 - intersection;

	return intersection / u;
}

template float GetOverlap(const vector<float>& bbox1, const vector<float>& bbox2);
template double GetOverlap(const vector<double>& bbox1, const vector<double>& bbox2);

template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12,
		const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22,const OverlapType overlap_type)
{
	Dtype x1 = MAX(x11, x21);
	Dtype y1 = MAX(y11, y21);
	Dtype x2 = MIN(x12, x22);
	Dtype y2 = MIN(y12, y22);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(x11, y11, x12, y12);
	Dtype area2 = GetArea(x21, y21, x22, y22);
	Dtype u = 0;
	switch(overlap_type)
	{
		case caffe::OVERLAP_UNION:
		{
			u = area1 + area2 - intersection;
			break;
		}
		case caffe::OVERLAP_BOX1:
		{
			u = area1 ;
			break;
		}
		case caffe::OVERLAP_BOX2:
		{
			u = area2 ;
			break;
		}
		default:
			LOG(FATAL) << "Unknown type " << overlap_type;
	}

	return intersection / u;
}
template float GetOverlap(const float x11, const float y11, const float x12, const float y12,
		 const float x21, const float y21, const float x22, const float y22,const OverlapType overlap_type);
template double GetOverlap(const double x11, const double y11, const double x12, const double y12,
		 const double x21, const double y21, const double x22, const double y22,const OverlapType overlap_type);


template <typename Dtype>
Dtype GetNMSOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2) {
	Dtype x1 = MAX(bbox1[0], bbox2[0]);
	Dtype y1 = MAX(bbox1[1], bbox2[1]);
	Dtype x2 = MIN(bbox1[2], bbox2[2]);
	Dtype y2 = MIN(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return 0.0;

	Dtype area2 = GetArea(bbox2);
	return w * h / area2;
}

template float GetNMSOverlap(const vector<float>& bbox1, const vector<float>& bbox2);
template double GetNMSOverlap(const vector<double>& bbox1, const vector<double>& bbox2);



vector<bool> GetPredictedResult(const vector< std::pair<int, vector<float> > > &gt_instances,
		const vector< std::pair<float, vector<float> > > &pred_instances, float ratio){
	vector<bool> res;
	vector<bool> used_gt_instance ;
	used_gt_instance.resize(gt_instances.size(),false);
	for(int pred_id = 0 ; pred_id < pred_instances.size(); pred_id ++){
		float max_overlap = 0;
		float used_id = -1;
		for(int gt_id = 0; gt_id < gt_instances.size(); ++ gt_id)
		{
			float overlap = GetOverlap(pred_instances[pred_id].second,gt_instances[gt_id].second);
			if( overlap >  max_overlap)
			{
				max_overlap = overlap;
				used_id = gt_id;
			}
		}
		if(used_id != -1 && max_overlap >= ratio)
		{
			res.push_back( used_gt_instance[used_id] == false);
			used_gt_instance[used_id] = true;
		}
		else
		{
			res.push_back(false);
		}
	}
	return res;
}



float GetTPFPPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& tpr,vector<float> &fpr){

	std::stable_sort(pred_instances_with_gt.begin(), pred_instances_with_gt.end(),
			compareCandidate<float>);
	tpr.clear();
	fpr.clear();
	int tp = 0;
	int negative_count = 0;
	for(int i=0; i< pred_instances_with_gt.size(); i++){
		negative_count += int(pred_instances_with_gt[i].second[4]) == 0 ? 1:0;
	}



	for(int i=0; i< pred_instances_with_gt.size(); i++)
	{
		tp += int(pred_instances_with_gt[i].second[4]) == 1 ? 1:0;
		fpr.push_back ((i+1 - tp)/(0.0 + negative_count) );
		tpr.push_back(tp/(0.0 + n_positive));

	}
	float auc = tpr[0]*fpr[0];
	for(int i=1; i< pred_instances_with_gt.size(); i++)
	{
		auc += tpr[i]*(fpr[i]-fpr[i-1]);
	}
	return auc;
}





float GetPRPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		const int n_positive, vector<float>& precision,vector<float> &recall){

	std::stable_sort(pred_instances_with_gt.begin(), pred_instances_with_gt.end(),
			compareCandidate<float>);
	precision.clear();
	recall.clear();
	int corrected_count = 0;
	for(int i=0; i< pred_instances_with_gt.size(); i++)
	{
		corrected_count += int(pred_instances_with_gt[i].second[4]) == 1 ? 1:0;
		precision.push_back(corrected_count/(i+0.0+1));
		recall.push_back(corrected_count/(0.0 + n_positive));

	}
	float ap = precision[0]*recall[0];
	for(int i=1; i< pred_instances_with_gt.size(); i++)
	{
		ap += precision[i]*(recall[i]-recall[i-1]);
	}
	return ap;
}




void GetPredictedWithGT_FDDB(const string gt_file, const string pred_file,
		vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
		int & n_positive, bool showing, string img_folder, string output_folder,float ratio  )
{
	FILE* gt_fd = NULL;
	FILE* pred_fd = NULL;
	gt_fd = fopen(gt_file.c_str(),"r");
	CHECK(gt_fd != NULL)<<" can not find gt_file "<<gt_file;
	pred_fd  = fopen(pred_file.c_str(),"r");
	CHECK(pred_fd != NULL) <<" can not find pred_file "<<pred_file;
	char img_name[255];
	char pred_img_name[255];
	n_positive = 0;
	pred_instances_with_gt.clear();
	cv::Mat src_img;
	char scores_c[100];
	while(fscanf(gt_fd,"%s",img_name) == 1)
	{
		if(showing)
		{

			src_img = cv::imread(img_folder +string(img_name)+string(".jpg"), CV_LOAD_IMAGE_COLOR);
			if (!src_img.data)
			{
				LOG(ERROR) << "Could not open or find file " <<
						img_folder +string(img_name)+string(".jpg");
			}

		}

		CHECK(fscanf(pred_fd,"%s",pred_img_name) == 1 && strcmp(pred_img_name,img_name) == 0);
		int n_face = 0;
		CHECK(fscanf(gt_fd,"%d",&n_face) == 1);
		//LOG(INFO)<<"image:"<<img_folder +string(img_name)+string(".jpg") <<"has " <<n_face<<" face";

		vector< std::pair<int, vector<float> > > gt_instances;
		vector< std::pair<int, vector<float> > > gt_instances_nohard;
		vector< std::pair<float, vector<float> > > pred_instances;

		for(int i=0; i < n_face; i++)
		{
			/**
			 * read faces in one image for gt
			 */
			float lt_x, lt_y, height,width;
			int label;
			CHECK(fscanf(gt_fd, "%f %f %f %f %d",&lt_x, &lt_y, &width, &height, &label) == 5);
			vector<float> temp;
			temp.push_back(lt_x);
			temp.push_back(lt_y);
			temp.push_back(lt_x+width);
			temp.push_back(lt_y+height);
			gt_instances.push_back( std::make_pair(label, temp));
			if(label == 0)
			{
				gt_instances_nohard.push_back(std::make_pair(label, temp));
				if(showing)
				{
					cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
							cv::Point(temp[2], temp[3]), cv::Scalar(255, 0, 0),1);

//					cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
//										cv::Point(temp[2], temp[3]), cv::Scalar(0, 0, 255),2);

				}
				n_positive += 1;
			}
		}

		CHECK(fscanf(pred_fd,"%d",&n_face) == 1);
		for(int i=0; i < n_face; i++)
		{
			float lt_x, lt_y, height,width,score;
			/**
			 * read faces in one image for gred
			 */
			vector<float> temp;
			temp.clear();
			CHECK(fscanf(pred_fd, "%f %f %f %f %f",&lt_x, &lt_y, &width, &height, &score) == 5);
			temp.push_back(lt_x);
			temp.push_back(lt_y);
			temp.push_back(lt_x+width);
			temp.push_back(lt_y+height);
			pred_instances.push_back(std::make_pair(score,temp));

		}

		vector<bool> corrected = GetPredictedResult(gt_instances,pred_instances,ratio);
		vector<bool> corrected_nohard = GetPredictedResult(gt_instances_nohard,pred_instances,ratio );
		for(int i=0; i < n_face; i++)
		{
			if( corrected[i] != corrected_nohard[i])
			{
				LOG(INFO)<<"Detected a hard sample";
				continue;
			}

			if(showing )
			{
				vector<float> temp = pred_instances[i].second;
//				cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
//						cv::Point(temp[2], temp[3]), cv::Scalar(255, 0, 0));
				if(corrected[i]){
					cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
										cv::Point(temp[2], temp[3]), cv::Scalar(0, 255, 0),1);
				}else{
					cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
										cv::Point(temp[2], temp[3]), cv::Scalar(0, 0, 255),1);
				}
//				cv::rectangle(src_img, cv::Point(temp[0], temp[1]),
//										cv::Point(temp[2], temp[3]), cv::Scalar(0, 0, 255),2);
				sprintf(scores_c, "%.3f", pred_instances[i].first);
				cv::putText(src_img, scores_c, cv::Point(temp[0]/2 + temp[2]/2, temp[1]),
					CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255));
			}

			pred_instances[i].second.push_back( corrected[i] == true? 1:0);
			pred_instances_with_gt.push_back(pred_instances[i]);
		}
		if(showing)
		{
			LOG(INFO)<<"saving :"<<output_folder +string(img_name)+string(".jpg");
			cv::imwrite(output_folder +string(img_name)+string(".jpg"), src_img);
		}
	}

	fclose(pred_fd);
	fclose(gt_fd);
}

cv::Scalar GetColorById(int id){
	return cv::Scalar(MAX(0.1, ((78*(id+1))%255)/255.0)*255,
			MAX(0.1, ((121*(id+1))%255)/255.0)*255,
			(MAX(0.1, ((42*(id+1))%255)/255.0)* 255));
}

void ShowClassColor(vector<string> class_names, string& out_name){
	int num_class = class_names.size();
	int pad_h = 20;
	int pad_w = 10;
	int rect_w = 30;
	int rect_h = 10;
	int thickness = 3;

	int total_h = num_class*(rect_h+pad_h)+pad_h;
	int total_w = 2*pad_w + rect_w +100;

	cv::Mat img = cv::Mat(total_h,total_w, CV_8UC3);
	for(int i=0; i< num_class; ++i){
		cv::rectangle(img, cv::Point(pad_w,i*(rect_h+pad_h)+pad_h ),
			cv::Point(pad_w+rect_w,i*(rect_h+pad_h)+pad_h + rect_h ),
			GetColorById(i),thickness);
		cv::putText(img, class_names[i], cv::Point(2*pad_w + rect_w, i*(rect_h+pad_h)+pad_h + rect_h/2 ),
				CV_FONT_HERSHEY_PLAIN, 1, GetColorById(i));
	}
	char path[256];
	sprintf(path, "%s.jpg",out_name.c_str() );
	cv::imwrite(string(path), img);

}

template <typename Dtype>
bool ShowMultiClassBBoxOnImage(const string img_path, const vector< vector< BBox<Dtype> > >& multiclass_bboxes,
		const vector<Dtype> multiclass_threshold,const string out_path,  int thickness,
		BufferedColorJPGReader<Dtype>* p_buffered_jpg_reader){
	cv::Mat src_img;
	if(p_buffered_jpg_reader == NULL){
		src_img = cv::imread(img_path+string(".jpg"), CV_LOAD_IMAGE_COLOR);
		if(! src_img.data)
		{
			src_img = cv::imread(img_path+string(".JPG"), CV_LOAD_IMAGE_COLOR);
			if(! src_img.data)
			{
                        src_img = cv::imread(img_path+string(".png"), CV_LOAD_IMAGE_COLOR);
                        if(! src_img.data){
				LOG(ERROR)<< "Could not open or find file " << img_path+string(".JPG");
				return false;}
			}
		}
	}else{
		p_buffered_jpg_reader->LoadToCvMat(img_path,src_img);
		if(! src_img.data)
		{
			LOG(ERROR)<< "Could not open or find file " << img_path;
			return false;
		}
	}
	int num_class  = multiclass_bboxes.size();
	CHECK_EQ(num_class, multiclass_threshold.size());
	for(int class_id = 0; class_id < num_class; ++class_id){
		ShowBBoxOnMat(src_img,  multiclass_bboxes[class_id], multiclass_threshold[class_id],
				GetColorById(class_id), thickness);
	}
	char path[256];
	sprintf(path, "%s.jpg",out_path.c_str() );
	cv::imwrite(string(path), src_img);
	return true;
}
template bool ShowMultiClassBBoxOnImage(const string img_path, const vector< vector< BBox<float> > >& multiclass_bboxes,
		const vector<float> multiclass_threshold,const string out_path,  int thickness,
		BufferedColorJPGReader<float>* p_buffered_jpg_reader);

template bool ShowMultiClassBBoxOnImage(const string img_path, const vector< vector< BBox<double> > >& multiclass_bboxes,
		const vector<double> multiclass_threshold,const string out_path,  int thickness,
		BufferedColorJPGReader<double>* p_buffered_jpg_reader);


template <typename Dtype>
bool ShowBBoxOnImage(const string img_path, const vector< BBox<Dtype> >& bboxes,
		const Dtype threshold,const string out_path,const cv::Scalar color, const int thickness,
		BufferedColorJPGReader<Dtype>* p_buffered_jpg_reader){
	cv::Mat src_img;
	if(p_buffered_jpg_reader == NULL){
		src_img = cv::imread(img_path+string(".jpg"), CV_LOAD_IMAGE_COLOR);
		if(! src_img.data)
		{
			src_img = cv::imread(img_path+string(".JPG"), CV_LOAD_IMAGE_COLOR);
			if(! src_img.data)
			{
				LOG(ERROR)<< "Could not open or find file " << img_path+string(".JPG");
				return false;
			}
		}
	}else{
		p_buffered_jpg_reader->LoadToCvMat(img_path,src_img);
		if(! src_img.data)
		{
			LOG(ERROR)<< "Could not open or find file " << img_path;
			return false;
		}
	}

	char path[256];
	ShowBBoxOnMat(src_img,  bboxes,  threshold, color, thickness);
	sprintf(path, "%s.jpg",out_path.c_str() );
	//LOG(INFO)<<"saving candidate image for "<< filename<<" to "<<string(path);
	cv::imwrite(string(path), src_img);
	return true;
}

template <typename Dtype>
void  ShowBBoxOnMat(cv::Mat& img,const vector< BBox<Dtype> >& bboxes,const Dtype threshold,
		const cv::Scalar color, const int thickness){
	char path[128];
	int additional_data_dim = 0;
	if(bboxes.size()>0){
		additional_data_dim = bboxes[0].data.size();
		CHECK_EQ(additional_data_dim % 3,0);
	}
	int n_additional_point = additional_data_dim / 3;
	vector<Dtype> point_data(n_additional_point*2,0);
	for(int candidate_id = 0; candidate_id < bboxes.size(); ++candidate_id){
		Dtype score = bboxes[candidate_id].score ;
		if(score < threshold)
			continue;

		vector<Dtype> crop_coords(4, 0);

		crop_coords[0] = MIN(img.cols, MAX(0, bboxes[candidate_id].x1));
		crop_coords[1] = MIN(img.rows, MAX(0, bboxes[candidate_id].y1));
		crop_coords[2] = MIN(img.cols, MAX(0, bboxes[candidate_id].x2));
		crop_coords[3] = MIN(img.rows, MAX(0, bboxes[candidate_id].y2));

		cv::rectangle(img, cv::Point(crop_coords[0], crop_coords[1]),
				cv::Point(crop_coords[2], crop_coords[3]), color, thickness);
		sprintf(path, "%.3f", score);
		cv::putText(img, path, cv::Point(crop_coords[0]/2 + crop_coords[2]/2, crop_coords[1]),
								CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255));
		CHECK_EQ(additional_data_dim ,bboxes[candidate_id].data.size());
		for(int p_id = 0; p_id < n_additional_point; ++p_id){
			point_data[p_id*2] = bboxes[candidate_id].data[n_additional_point+p_id*2];
			point_data[p_id*2+1] = bboxes[candidate_id].data[n_additional_point+p_id*2+1];
			cv::circle( img , cv::Point( point_data[p_id*2],point_data[p_id*2+1]), 2, caffe::GetColorById(p_id));
		}
		if(n_additional_point == 8){
			vector<pair<int,int> > point_line;
			point_line.push_back(pair<int, int>(0,1));
			point_line.push_back(pair<int, int>(1,3));
			point_line.push_back(pair<int, int>(2,3));
			point_line.push_back(pair<int, int>(2,0));
			point_line.push_back(pair<int, int>(4,5));
			point_line.push_back(pair<int, int>(5,7));
			point_line.push_back(pair<int, int>(7,6));
			point_line.push_back(pair<int, int>(4,6));
			point_line.push_back(pair<int, int>(2,4));
			point_line.push_back(pair<int, int>(3,5));
			point_line.push_back(pair<int, int>(1,7));
			point_line.push_back(pair<int, int>(0,6));

			for(int line_id =0; line_id < point_line.size();++line_id){
				int point_1 = point_line[line_id].first;
				int point_2 = point_line[line_id].second;
				cv::line(img , cv::Point( point_data[point_1*2],point_data[point_1*2+1]) ,
						cv::Point( point_data[point_2*2],point_data[point_2*2+1]), caffe::GetColorById(line_id/4),2);
			}
		}

//		cv::line(mat,cv::Point( round(cur_point_1.x), round(cur_point_1.y)),
//						cv::Point( round(cur_point_2.x), round(cur_point_2.y)), color,thickness);


	}
	return ;
}

template void ShowBBoxOnMat(cv::Mat& img,const vector< BBox<float> >& bboxes,const float threshold,
		const cv::Scalar color, const int thickness);

template void ShowBBoxOnMat(cv::Mat& img,const vector< BBox<double> >& bboxes,const double threshold,
		const cv::Scalar color, const int thickness);

template bool ShowBBoxOnImage(const string img_path, const vector< BBox<double> >& bboxes,
				const double threshold,const string out_path,const cv::Scalar color, const int thickness,
				BufferedColorJPGReader<double>* p_buffered_jpg_reader);
template bool ShowBBoxOnImage(const string img_path, const vector< BBox<float> >& bboxes,
				const float threshold,const string out_path,const cv::Scalar color, const int thickness,
				BufferedColorJPGReader<float>* p_buffered_jpg_reader);

template <typename Dtype>
void PushBBoxTo(std::ofstream & out_result_file,const vector< BBox<Dtype> >& bboxes){
	int additional_data_dim = 0;
	if(bboxes.size()>0){
		additional_data_dim = bboxes[0].data.size();
	}
	for(int candidate_id=0; candidate_id < bboxes.size(); ++candidate_id){
		out_result_file << bboxes[candidate_id].x1<<" "<<
				bboxes[candidate_id].y1 << " "<<bboxes[candidate_id].x2 - bboxes[candidate_id].x1 <<" "<<
				bboxes[candidate_id].y2 - bboxes[candidate_id].y1<<" "<<
				bboxes[candidate_id].score<<" ";
		CHECK_EQ(bboxes[candidate_id].data.size(), additional_data_dim);
		for(int id=0; id<additional_data_dim; ++id ){
			out_result_file << bboxes[candidate_id].data[id]<<" ";
		}
		out_result_file<<std::endl;
	}
}

template void PushBBoxTo(std::ofstream & out_result_file,const vector< BBox<float> >& bboxes);
template void PushBBoxTo(std::ofstream & out_result_file,const vector< BBox<double> >& bboxes);

std::vector<std::string> std_split(const std::string& stri,const std::string& pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	string str= stri+pattern;
	int size=str.size();

	for(int i=0; i<size; i++)
	{
		pos=str.find(pattern,i);
		if(pos<size)
		{
		  std::string s=str.substr(i,pos-i);
		  result.push_back(s);
		  i=pos+pattern.size()-1;
		}
	}
	return result;

}


template <typename Dtype>
void DrawScorePointsOnMat(cv::Mat &mat, const vector<ScorePoint<Dtype> >& points, const cv::Scalar &color,
		const int r,const vector<int>& line_point_pairs, int thickness){
	CHECK_EQ(line_point_pairs.size()%2, 0);
	const int n_point = points.size();
	const int n_pairs = line_point_pairs.size()/2;
	for(int i=0; i < n_point; ++i){
		const ScorePoint<Dtype>& cur_point = points[i];
		cv::circle(mat, cv::Point( round(cur_point.x), round(cur_point.y)), r, color);
	}
	for(int i=0; i < n_pairs; ++i){
		CHECK_LT(line_point_pairs[i*2],   n_point);
		CHECK_LT(line_point_pairs[i*2+1], n_point);
		const ScorePoint<Dtype>& cur_point_1 = points[line_point_pairs[i*2]];
		const ScorePoint<Dtype>& cur_point_2 = points[line_point_pairs[i*2+1]];
		if(cur_point_1.score > 0 && cur_point_2.score > 0 ){
			cv::line(mat,cv::Point( round(cur_point_1.x), round(cur_point_1.y)),
				cv::Point( round(cur_point_2.x), round(cur_point_2.y)), color,thickness);
		}
	}
}

template void DrawScorePointsOnMat(cv::Mat &mat,const vector<ScorePoint<float> >& points, const cv::Scalar &color,
		const int r,const vector<int>& line_point_pairs, int thickness);
template void DrawScorePointsOnMat(cv::Mat &mat,const vector<ScorePoint<double> >& points, const cv::Scalar &color,
		const int r,const vector<int>& line_point_pairs, int thickness);

INSTANTIATE_STRUCT(BBox);
INSTANTIATE_STRUCT(ScorePoint);
}
// namespace caffe

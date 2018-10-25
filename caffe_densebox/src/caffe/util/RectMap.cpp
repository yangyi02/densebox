/*
 * RectMap.cpp
 *
 *  Created on: 2015年5月8日
 *      Author: Alan_Huang
 */
#include "caffe/util/RectMap.hpp"
#include <climits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifndef MAX_MAP_SIZE
#  define MAX_MAP_SIZE  1000000
#endif



RectPoint::RectPoint(){
	x=y=0;
}

RectPoint::~RectPoint(){

}
RectPoint::RectPoint(int y_,int x_){
	x = x_;
	y = y_;
}

bool RectPoint::operator < (const RectPoint& other)const{
	if(y < other.y)
		return true;
	else
		return x < other.x;
}

RectPoint RectPoint::add(int dy,int dx) const {
	return RectPoint(y+dy,x+dx);
}

// ####################For Rect ###################
Rect::Rect(){
	left_top = RectPoint();
	height = width = 0;
}

Rect::Rect(RectPoint left_top_, int height_, int width_){
	left_top = left_top_;
	height = height_;
	width = width_;
}
Rect::~Rect(){

}
bool Rect::Contain(const RectPoint&  point) const{
	return (point.x >= left_top.x && point.x < left_top.x + width &&
		point.y >= left_top.y && point.y < left_top.y + height );
}

bool Rect::Contain(const Rect& other) const{
	return (Contain(other.left_top) &&
			Contain(other.left_top.add(other.height-1, other.width -1)));
}

int Rect::Overlap(const Rect&  other){
	int x1 = MAX(left_top.x, other.left_top.x);
	int y1 = MAX(left_top.y, other.left_top.y);
	int x2 = MIN(left_top.x + width,  other.left_top.x + other.width);
	int y2 = MIN(left_top.y + height, other.left_top.y + other.height);
	if( (x2 - x1 )<= 0 || (y2 - y1) <= 0)
		return 0;
	return MAX((x2 - x1 ) * (y2 - y1),0);
}

Rect Rect::MoveBy(int dy, int dx)const{
	Rect rect;
	rect.height = height;
	rect.width = width;
	rect.left_top = left_top.add(dy,dx);
	return rect;
}

Rect Rect::MoveBy(RectPoint dydx)const{
	return MoveBy(dydx.y,dydx.x);
}


// ####################For RectMap###################
RectMap::RectMap(){
	this->Clear();

}

RectMap::~RectMap(){

}
bool RectMap::Occupied(const RectPoint& point){
	for(int i=0; i < placedRects.size(); ++i){
		if(placedRects[i].Contain(point))
			return true;
	}
	return false;
}

int RectMap::MapHeight(){
	return *(horizontal_line.begin());
}

int RectMap::MapWidth(){
	return *(vertical_line.begin());
}

int RectMap::GetArea(){
	return MapWidth() * MapHeight();
}

int RectMap::GetArea(const Rect& added_rect){
	int height = MAX(MapHeight(), added_rect.left_top.y + added_rect.height);
	int width = MAX(MapWidth(), added_rect.left_top.x + added_rect.width);
	return height* width;
}


const vector<Rect>& RectMap::GetPlacedRects(){
	return placedRects;
}

map<RectPoint, int>& RectMap::GetCandidatePLeftTopPoints(){
	return candidateLeftTopPoint;
}


ostream& operator << (ostream & stream,Rect & rect){
	stream<< "("<<rect.left_top.x<<","<<rect.left_top.y<<","<<rect.width<<","<<rect.height<<")";
	return stream;
}



ostream& operator << (ostream & stream,RectPoint & point){
	stream<< "("<<point.x<<","<<point.y<<")";
	return stream;
}

void RectMap::PruneInvalidCandidatePoint(){
//	std::cout<<" PruneInvalidCandidatePoint start "<<std::endl;
	for(map<RectPoint,int>::iterator it = candidateLeftTopPoint.begin();
			it != candidateLeftTopPoint.end();){
		RectPoint point = it->first;

		if(Occupied(it->first))
			candidateLeftTopPoint.erase(it++);
		else
			++it;
	}
//	std::cout<<" PruneInvalidCandidatePoint end "<<std::endl;
}

int RectMap::TryToPlaceRectAt(const Rect& rect, RectPoint point){
	Rect rect_new =	Rect(point, rect.height,rect.width);
	for(int i=0; i < placedRects.size();++i){
		if(rect_new.Overlap(placedRects[i]) > 0 ){
			return INT_MAX;
		}
	}
	return GetArea(rect_new);
}

RectPoint RectMap::GreedyFindBestPointToPlace(const Rect& rect){
	int min_area = INT_MAX;
	RectPoint best_point = RectPoint();
	for(map<RectPoint,int>::iterator it = candidateLeftTopPoint.begin();
				it != candidateLeftTopPoint.end();++it){
		int cur_area = TryToPlaceRectAt(rect, it->first);
//		std::cout<<"area: "<<cur_area<<"  if place Rect " <<rect.height <<" "<<rect.width<<
//				" at the point("<<it->first.y<<","<<it->first.x<<"). "<<std::endl;
		if(cur_area < min_area){
			min_area = cur_area;
			best_point = it->first;
		}
	}
	return best_point;
}

bool RectMap::CheckNoOverlap(){
	for(int i = 0 ; i < placedRects.size(); ++i){
		for(int j = 0 ; j < placedRects.size(); ++j){
			if( i == j)
				continue;
			float overlap_ratio = placedRects[i].Overlap(placedRects[j]);
			std::cout<<"overlap ratio of "<< i <<"  "<< j <<"  : "<<overlap_ratio<<std::endl;
		}
	}
	return false;
}

void RectMap::PlaceCornerPoint(const RectPoint point){
	set<int>::iterator founded = vertical_line.find(point.x);
	if(founded == vertical_line.end()){
		vertical_line.insert(point.x);
		for(set<int>::iterator it = horizontal_line.begin(); it != horizontal_line.end(); ++it){
			candidateLeftTopPoint.insert(pair<RectPoint,int>(
					RectPoint(*it,point.x),0));
		}
	}
	founded = horizontal_line.find(point.y);
	if(founded == horizontal_line.end()){
		horizontal_line.insert(point.y);
		for(set<int>::iterator it = vertical_line.begin(); it != vertical_line.end(); ++it){
			candidateLeftTopPoint.insert(pair<RectPoint,int>(
					RectPoint(point.y,*it),0));
		}
	}
}

bool RectMap::PlaceRect(const Rect& rect){
	RectPoint point_to_place = GreedyFindBestPointToPlace(rect);
	Rect rect_new =	Rect(point_to_place, rect.height,rect.width);
	PlaceCornerPoint(point_to_place);
	PlaceCornerPoint(point_to_place.add(rect.height,rect.width));
	placedRects.push_back(rect_new);
	placedRectIds.push_back(placedRectIds.size());
	PruneInvalidCandidatePoint();
	return true;
}

int RectMap::GetRectId(const int y, const int x){

	for(int i = 0; i < placedRects.size(); ++i){
		Rect cur_rect = placedRects[i];
		if( x >= cur_rect.left_top.x && x < (cur_rect.left_top.x + cur_rect.width) &&
				y >= cur_rect.left_top.y && y < (cur_rect.left_top.y + cur_rect.height))
			return i;
	}
	return -1;
}

void RectMap::Clear(){
	placedRects.clear();
	placedRectIds.clear();

	horizontal_line.clear();
	horizontal_line.insert(0);
	vertical_line.clear();
	vertical_line.insert(0);
	candidateLeftTopPoint.clear();
	candidateLeftTopPoint.insert(pair<RectPoint,int>(RectPoint(),0));
}

// #################### For RectMapPainter ###################

RectMapPainter::RectMapPainter(int map_height,int map_width){
	InitPannel(map_height, map_width);
}

void RectMapPainter::InitPannel(int map_height,int map_width)
{
	pannel = cv::Mat::zeros(map_height, map_width,CV_8UC3);
}
RectMapPainter::~RectMapPainter(){

}

vector<float> RectMapPainter::GetColorById(int id){
	vector<float> color ;
	color.push_back(MAX(0.1, ((78*(id+1))%255)/255.0));
	color.push_back(MAX(0.1, ((121*(id+1))%255)/255.0));
	color.push_back(MAX(0.1, ((42*(id+1))%255)/255.0));
	return color;
}

void RectMapPainter::DrawRect(const Rect& rect, int id){
	vector<float> color = GetColorById(id);
	int start_h = rect.left_top.y;
	int start_w = rect.left_top.x;
	int end_h = start_h + rect.height;
	int end_w = start_w + rect.width;
	if(end_h > pannel.rows || end_w > pannel.cols){
		std::cout<<" rect exceed the range of pannel!"<<std::endl;
		return;
	}
	for(int  h = start_h; h < end_h; ++h){
		for(int w = start_w; w < end_w; ++w){
			for(int c = 0; c < 3 ; ++c){
				pannel.at<cv::Vec3b>(h, w)[c] = static_cast<uint8_t>( 255 * color[c] );
			}
		}
	}
}

void RectMapPainter::DrawPoints(  map<RectPoint, int>& candidateLeftTopPoint){
	for(map<RectPoint,int>::iterator it = candidateLeftTopPoint.begin();
					it != candidateLeftTopPoint.end();++it){
		RectPoint point = it->first;
		cv::circle(pannel,cv::Point(point.x,point.y),2, cv::Scalar(255, 255, 255));
	}

}

void RectMapPainter::DrawRects(const vector<Rect>& rects, int start_id  ){
	for(int i=0 ; i < rects.size();++i){
		DrawRect(rects[i], i + start_id);
	}
}

void RectMapPainter::SaveImg( string  img_name){
	imwrite(img_name, pannel);
}

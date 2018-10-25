/*
 * RectMap.hpp
 *
 *  Created on: 2015年5月8日
 *      Author: Alan_Huang
 */

#ifndef RECTMAP_HPP_
#define RECTMAP_HPP_

#include<vector>
#include<map>
#include<string>
#include<set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class RectPoint{
public:
	int x,y;
	RectPoint();
	~RectPoint();
	RectPoint(int y_,int x_);
	bool operator<(const RectPoint& other)const;
	RectPoint add(int dy,int dx) const;
	friend ostream& operator << (ostream & stream,RectPoint & point);
};


class Rect{
public:
	Rect();
	~Rect();
	friend ostream& operator << (ostream & stream,Rect & rect);
	Rect(RectPoint left_top_,int height_ = 0, int width_ = 0);
	bool Contain(const RectPoint&  point) const;
	bool Contain(const Rect& other) const ;
	int  Overlap(const Rect& other) ;
	Rect MoveBy(int dy, int dx) const;
	Rect MoveBy(RectPoint dydx) const;
	RectPoint left_top;
	int height,width;
};

class RectMap{
public:
	RectMap();
	~RectMap();
	bool Occupied(const RectPoint& point);
	int MapHeight();
	int MapWidth();
	int GetArea();
	const vector<Rect>& GetPlacedRects();
	map<RectPoint, int>& GetCandidatePLeftTopPoints();
	bool PlaceRect(const Rect& rect);
	bool CheckNoOverlap();
	int GetRectId(const int y, const int x);
	void Clear();
private:
	/**
	 * Return the total area if place rectangle at this point.
	 * If cannot place the rectange here, return INT_MAX;
	 */
	int TryToPlaceRectAt(const Rect& rect, RectPoint point);
	/**
	 * return the area if the added_rect is placed
	 */
	int GetArea(const Rect& added_rect);
	RectPoint GreedyFindBestPointToPlace(const Rect& rect);
	/**
	 * place a corner point in map, which updates the
	 * horizontal_line, vertical_line, candidateLeftTopPoint;
	 */
	void PlaceCornerPoint(const RectPoint point);
	void PruneInvalidCandidatePoint();
//	void PrintPoints(map<RectPoint, int>& candidateLeftTopPoint);
	vector<Rect> placedRects;
	vector<int> placedRectIds;
	// line horizontal, contain y axis， largest first
	set<int,greater<int> > horizontal_line;
	// line vertical, contain x axis， largest first
	set<int,greater<int> > vertical_line;
	map<RectPoint, int> candidateLeftTopPoint; // point, id

};

class RectMapPainter{
public:
	RectMapPainter(int map_height = 5000, int map_width = 5000);
	~RectMapPainter();
	void InitPannel(int map_height = 5000, int map_width = 5000);
	void DrawRect(const Rect& rect, int id);
	void DrawRects(const vector<Rect>& rects, int start_id = 0 );
	void DrawPoints(  map<RectPoint, int>& candidateLeftTopPoint);
	void SaveImg(  string  img_name);
private:
	vector<float> GetColorById(int id);
	cv::Mat pannel;
};

#endif /* RECTMAP_HPP_ */

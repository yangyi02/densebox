/*
 * Main.cpp
 *
 *  Created on: 2015年5月11日
 *      Author: Alan_Huang
 */
#include "caffe/util/RectMap.hpp"
#include <cmath>
#include <iostream>
using namespace std;
int main(){

	RectMap map;

	float scale_start = 1;
	float scale_end = -3;
	float scale_step = -0.45;
	int img_h = 600;
	int img_w = 800;
//	int rect_id = 0;
	for(float scale = scale_start; scale >= scale_end; scale+=scale_step){
		float cur_scale = pow(2,scale);
		Rect rect = Rect(RectPoint(), int(img_h * cur_scale), int(img_w * cur_scale));
		map.PlaceRect(rect);
//		std::cout<<"scale "<<cur_scale<<" finished for rect_id = "<<rect_id++<<std::endl;
	}
	map.CheckNoOverlap();
	RectMapPainter map_painter = RectMapPainter(map.MapHeight(),map.MapWidth());
	std::cout<<"map.MapHeight():"<<map.MapHeight()<<"  map.MapWidth():"<<map.MapWidth()<<std::endl;
	map_painter.DrawRects(map.GetPlacedRects());
	map_painter.DrawPoints(map.GetCandidatePLeftTopPoints());
	map_painter.SaveImg("map.jpg");
	return 1;
}




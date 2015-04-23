/*
 * ObjectDetector.h
 *
 *  Created on: Sep 5, 2014
 *      Author: davy
 */

#ifndef OBJECTDETECTOR_H_
#define OBJECTDETECTOR_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


class ObjectDetector {

public:
	virtual void detectBiggestObject(const Mat &img, Rect &object,int DETECTION_WIDTH);
	virtual void initDetector(const char *cascadeFilename);
protected:
	CascadeClassifier objectDetector;
	int DETECTION_WIDTH;

};

#endif /* OBJECTDETECTOR_H_ */

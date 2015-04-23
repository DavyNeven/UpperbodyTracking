/*
 * Track.h
 * Author: davy
 */

#ifndef LKT_TRACKER_H_
#define LKT_TRACKER_H_

#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std; 

#define MAX_DETECTION_COUNTER 100


const TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
const Size subPixWinSize(5,5),winSize(31,31);	

class LKT_Tracker{
	protected:
		bool locked; 
		int detectionCounter; 
		double contArea; 
		Mat gray, prevGray; 
		vector<Point2f> points[2]; 
		Point bb[4];  
	    int featureCount;

	public:
		LKT_Tracker(int featureCount);
		void updateTracker(Mat &img, Rect pos);
		Rect getRect();
		void track(Mat &img);
		bool isLocked(); 
		Point* getPosition();
		virtual ~LKT_Tracker();
};

#endif /* LKT_TRACKER_H_ */

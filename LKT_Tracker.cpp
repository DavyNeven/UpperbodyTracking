


#include "LKT_Tracker.h"

LKT_Tracker::LKT_Tracker(int featureCount)
{
	this->featureCount = featureCount;
	detectionCounter = 0; 
	contArea = 0; 
	locked = false; 
}

void LKT_Tracker::updateTracker(Mat &img, Rect pos)
{
	locked = true;
	detectionCounter = 0;
	
	// Convert image to grayscale
	cvtColor(img, prevGray, COLOR_BGR2GRAY);
	// Find good features to track
	Mat mask = Mat::zeros(480,640, CV_8U);
	mask(pos) = 1; 
	/*const TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
	const Size subPixWinSize(5,5),winSize(31,31);*/
	goodFeaturesToTrack(prevGray, points[0], featureCount, 0.01, 5, mask, 
							3, 0, 0.04);
	cornerSubPix(prevGray, points[0], subPixWinSize, Size(-1,-1),
							 termcrit);
	// Extract points from rectangle
	bb[0] = Point(pos.x, pos.y);
	bb[1] = Point(pos.x+pos.width, pos.y);
	bb[3] = Point(pos.x, pos.y+pos.height);
	bb[2] = Point(pos.x+pos.width, pos.y + pos.height);
	
	vector<Point2f> bbb; 
	bbb.push_back(bb[0]);bbb.push_back(bb[1]);bbb.push_back(bb[2]);bbb.push_back(bb[3]); 
	
	contArea = contourArea(bbb); 
	cout << "Area is: " << contArea << endl;

	rectangle(img, pos, cv::Scalar(0,0,255), 2);
	
	
}
void LKT_Tracker::track(Mat &img)
{	
	if(locked)
	{
		
		/*if(detectionCounter++ > MAX_DETECTION_COUNTER)
		{
			locked = false;
			cout << "Unlocking of tracker" << endl; 
		}*/
		// Convert image to grayscale
		cvtColor(img, gray, COLOR_BGR2GRAY);
		// In case prevGray is not locked
		if(prevGray.empty())
			gray.copyTo(prevGray);
		vector<uchar> status;
		vector<float> err;
		/*const TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
		const Size subPixWinSize(5,5),winSize(31,31);*/
		calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, 
													err, winSize,3, termcrit, 0, 0.001);
		int i, k;
		for(i = k = 0; i< points[1].size(); i++)
		{
			if(!status[i])
				continue;
			points[0][k] = points[0][i];
			points[1][k++] = points[1][i];
			circle(img, points[1][i], 3, Scalar(0,0,255), -1, 1);
		}

		points[0].resize(k);
		points[1].resize(k);
		if(k < 3) 
		{
			locked = false;
		} 
		
		vector<Point2f> bbb; 
		bbb.push_back(bb[0]);bbb.push_back(bb[1]);bbb.push_back(bb[2]);bbb.push_back(bb[3]); 
		
		//find transformation between features
		try
		{
			if(points[0].size() > 2)
			{ 
				Mat transf = estimateRigidTransform(points[0], points[1], false);
				transform(bbb, bbb, transf);
				bb[0] = bbb.at(0);bb[1] = bbb.at(1);bb[2] = bbb.at(2);bb[3] = bbb.at(3);
				line(img, bb[0], bb[1], Scalar(0,0,255), 1,8);
				line(img, bb[1], bb[2], Scalar(0,0,255), 1,8);	
				line(img, bb[2], bb[3], Scalar(0,0,255), 1,8);	
				line(img, bb[3], bb[0], Scalar(0,0,255), 1,8);		
			}
		}		
		catch( cv::Exception &e){};
		contArea = contourArea(bbb); 
		cout << "Area is: " << contArea << endl; 
		std::swap(points[1], points[0]);	
		cv::swap(prevGray, gray);
	}
}
Rect LKT_Tracker :: getRect()
{
	//find min x, max x, min y and max y;
	int minx = 1000; int miny = 1000; int maxx = 0; int maxy = 0;
	for(int i = 0; i< 4; i++)
	{
		if(bb[i].x > maxx)
			maxx = bb[i].x;
		if(bb[i].x < minx)
			minx = bb[i].x;
		if(bb[i].y > maxy)
			maxy = bb[i].y;
		if(bb[i].y < miny)
			miny = bb[i].y;
	}
	Rect rect(minx, miny, maxy - miny, maxx-minx);
	return rect;

}
bool LKT_Tracker :: isLocked()
{
	return locked; 
}
Point* LKT_Tracker :: getPosition()
{
	return bb; 
} 
LKT_Tracker:: ~LKT_Tracker()
{

}


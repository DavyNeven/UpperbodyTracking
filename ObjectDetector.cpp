/*
 * ObjectDetector.cpp
 *
 *  Created on: Sep 5, 2014
 *      Author: davy
 */

#include "ObjectDetector.h"

void ObjectDetector::detectBiggestObject(const Mat &img,Rect &object, int DETECTION_WIDTH)
{
		// Shrink the image, to run faster
		Mat smallImg;
		float scale = img.cols / (float) DETECTION_WIDTH;
		if (img.cols > DETECTION_WIDTH) {
			// Shrink image while keeping same aspect ratio.
			int scaledHeight = cvRound(img.rows/scale);
			resize(img, smallImg, Size(DETECTION_WIDTH, scaledHeight));
		}
		else {
			// img already small
			smallImg = img;
		}

		Mat gray;
		if(smallImg.channels() == 3){
			cvtColor(smallImg, gray, CV_BGR2GRAY);
		}
		else if (img.channels() == 4) {
			cvtColor(smallImg, gray, CV_BGRA2GRAY);
		}
		else {
			// Access grayscale input directly
			gray = smallImg;
		}

		Mat equalizedImg;
		equalizeHist(gray, equalizedImg);

		//Detection
		int flags = CASCADE_FIND_BIGGEST_OBJECT; //Search biggest object
		Size minFeatureSize(80,80); 	//Smallest object Size
		float searchScaleFactor = 1.1f;
		int minNeighbors = 1;  	//Reliability vs many objects

		vector<Rect> objects;
		objectDetector.detectMultiScale(equalizedImg, objects,searchScaleFactor,
					minNeighbors,flags, minFeatureSize);

		if(objects.size() > 0) //There is an object detected
		{
				object = objects[0];
				// Enlarge if the image was shrunk before detection.
				if (img.cols > DETECTION_WIDTH)
				{
					//for (int i = 0; i < (int)objects.size(); i++ )
					//{
						object.x = cvRound(object.x * scale);
						object.y = cvRound(object.y * scale);
						object.width = cvRound(object.width * scale);
						object.height = cvRound(object.height * scale);
					//}
				}
				// Make sure the object is completely within the image, in case it was on a border.
				//for (int i = 0; i < (int)objects.size(); i++ )
				//{
					if (object.x < 0)
						object.x = 0;
					if (object.y < 0)
						object.y = 0;
					if (object.x + object.width > img.cols)
						object.x = img.cols - object.width;
					if (object.y + object.height > img.rows)
						object.y = img.rows - object.height;
				//}
		}
}
void ObjectDetector::initDetector(const char* cascadeFilename)
{
	try{
		objectDetector.load(cascadeFilename);
	}
	catch (cv::Exception e){}
	if (objectDetector.empty())
	{
		cerr << "ERROR: Couldn't load Face Detector(";
		cerr << cascadeFilename << ")!" << endl;
		exit(1);
	}
	this->DETECTION_WIDTH = DETECTION_WIDTH;
}


//
// Created by davy on 4/13/15.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include "LKT_Tracker.h"

using namespace cv;
using namespace std;


const char *faceCascadeFilename1 = "../haarcascade_upperbody.xml";
const char *faceCascadeFilename2 = "../HS.xml";
const int DETECTION_WIDTH = 640;

void initWebcam(cv::VideoCapture &cap, int cameraNumber,int width, int height)
{
    cap.open(cameraNumber);
    if(!cap.isOpened())
    {
        cerr << "ERROR: could not acces this camera or video!" << endl;
        exit(1);
    }
    //Try to set camera resolution

    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
}

enum STATES {UB_DECTECTION=0, FACE_TRACKING, UB_TRACKING};
const char* STATE_NAMES[] = {"UpperBody detection", "Face tracking",
                                "UpperBody tracking"};

STATES state = UB_DECTECTION;

void startTracking(VideoCapture &videoCapture)
{
    Mat webcamImage;
    ObjectDetector bodyDetector1;
    ObjectDetector bodyDetector2;
    bodyDetector1.initDetector(faceCascadeFilename1);
    bodyDetector2.initDetector(faceCascadeFilename2);

    // Adding dlib facedetector
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // Initialize tracker
    LKT_Tracker *faceTracker = new LKT_Tracker(40);
    LKT_Tracker *UB_tracker = new LKT_Tracker(100);

    namedWindow("Upperbody");

    while(true)
    {
        videoCapture >> webcamImage;
        switch(state)
        {
            case UB_DECTECTION:
                            {
                                Rect rect;
                                bodyDetector1.detectBiggestObject(webcamImage, rect, 320);
                                cout << "Mode: UB_DETECTION" << endl;
                                if (rect.width > 0) // Upperbody detected!
                                {
                                    Rect rect1;
                                    bodyDetector2.detectBiggestObject(webcamImage,rect1,320);
                                    Rect intersect = rect1 & rect;
                                    if(intersect.area() > rect.area()/2)
                                    {
                                        //valid upperbody!
                                        //Start tracking!
                                        UB_tracker->updateTracker(webcamImage, rect1);
                                        state = UB_TRACKING;
                                    }
                                }
                            }
                                break;
            case FACE_TRACKING:
                            {
                            	cout << "Mode: face tracking" << endl;
                                faceTracker->track(webcamImage);
                                // Try to update the tracker with a face detection //will do this every 4th frame
                                static int counter = 0;
                                if(counter++%4 == 0)
                                {
                                    dlib::cv_image<dlib::bgr_pixel> cv(webcamImage);
                                    dlib::array2d<dlib::rgb_pixel> imgDlib;
                                    dlib::assign_image(imgDlib, cv);
                                    std::vector<dlib::rectangle> dets = detector(imgDlib);

                                    if (dets.size() > 0) { // Face detected -> update tracker
                                        Rect faceRect =
                                                Rect(dets[0].left(), dets[0].top(), dets[0].width(),
                                                     dets[0].height()) & Rect(0, 0, 640, 480);
                                        // Start tracker
                                        faceTracker->updateTracker(webcamImage, faceRect);
                                    }
                                }
                            }    break;
            case UB_TRACKING:
                            {
                            	cout << "Mode: UB_tracking" << endl;
                                //Try to update tracker;
                                Rect rect1;
                                bodyDetector2.detectBiggestObject(webcamImage,rect1,320);
                                if(((Rect)(rect1 & UB_tracker->getRect())).area() > UB_tracker->getRect().area()*0.8 && rect1.area() < UB_tracker->getRect().area()*1.2)
                                {
                                    UB_tracker->updateTracker(webcamImage, rect1);
                                }
                                else {
                                    UB_tracker->track(webcamImage);
                                }
                                // Try to update the tracker with a face detection //will do this every 4th frame
                                static int counter = 0;
                                if (counter++ % 4 == 0) {
                                    dlib::cv_image<dlib::bgr_pixel> cv(webcamImage);
                                    dlib::array2d<dlib::rgb_pixel> imgDlib;
                                    dlib::assign_image(imgDlib, cv);
                                    std::vector<dlib::rectangle> dets = detector(imgDlib);

                                    if (dets.size() > 0) { // Face detected -> update tracker
                                        Rect faceRect =
                                                Rect(dets[0].left(), dets[0].top(), dets[0].width(),
                                                     dets[0].height()) & Rect(0, 0, 640, 480);
                                        // Start tracker
                                        faceTracker->updateTracker(webcamImage, faceRect);
                                        state = FACE_TRACKING;
                                    }
                                }
                            } break;
        };
        imshow("Upperbody", webcamImage);
        if(waitKey(10) >= 0)
        {
            cout << "App finished" << endl;
            break;
        }

    }
}




int main(int argc, char** argv)
{

    VideoCapture cap;
    initWebcam(cap, 0, 640, 480);

    startTracking(cap);

    return 0;
}

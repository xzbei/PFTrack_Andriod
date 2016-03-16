#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "defs.h"
//#include "utils.h"
#include "particles.h"
#include "observation.h"
#include "time.h"
#include "unistd.h"
#include "opencv/cv.h"
//#include <stdio.h>
//#pragma comment (lib, "libgsl.a")
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
//#include <gsl/*.h>
#include <android/log.h>

#define PARTICLES 300
#define MAX_OBJECTS 1
#define U0 0.20
#define U1 0.20

#define MODE_RESET 0
#define MODE_BEGIN 1
#define MODE_TRAIN 2
#define MODE_TEST 5

//typedef struct params {
//    CvPoint loc1[MAX_OBJECTS];
//    CvPoint loc2[MAX_OBJECTS];
//    IplImage* objects[MAX_OBJECTS];
//    char* win_name;
//    IplImage* orig_img;
//    IplImage* cur_img;
//    int n;
//} params;

using namespace std;
using namespace cv;

int num_particles = PARTICLES;    /* number of particles */
int show_all = 1;             /* TRUE to display all particles */
IplImage* frame, * hsv_frame;
histogram** ref_histos;

particle* particles, * new_particles;
CvScalar color;
CvRect* regions;


extern "C" {
JNIEXPORT int JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba,int mode);

JNIEXPORT int JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba,int mode)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;

//    Mat& dst;
//    resize(mRgb, mRgb, Size(), 0.1, 0.1, INTER_CUBIC);

    int rows = mRgb.rows;
    int cols = mRgb.cols;
    int xcenter = rows/2;
    int ycenter = cols/2;
    int num_objects = 1;
    float s;
    int i, j, k, w, h, x, y,x1,y1,num,xcenter1,ycenter1,x0,y0,ww,hh;
    int c = 0;

    IplImage img = (IplImage)mRgb;
    switch (mode){
        case MODE_BEGIN:
            rectangle(mRgb,Point(ycenter-ycenter/2,xcenter - xcenter/2), Point(ycenter+ycenter/2,xcenter+xcenter/2),Scalar(255,0,0,255),8);
            __android_log_print(ANDROID_LOG_VERBOSE, "begin","rows  = %d",rows);
            __android_log_print(ANDROID_LOG_VERBOSE, "begin","cols  = %d",cols);
            return MODE_BEGIN;
            break;
        case MODE_RESET:
            return MODE_BEGIN;
            break;
        case MODE_TRAIN:
            frame = &img;
            hsv_frame = bgr2hsv(frame);
            CvRect* r;
            r = (CvRect*)malloc( 1 * sizeof( CvRect ) );
            xcenter = frame->width/2;
            ycenter = frame->height/2;

            x1 = round(xcenter - xcenter/2);
            y1 = round(ycenter - ycenter/2);

            ww = xcenter /2;
            hh = ycenter / 2;

            ww = ( ww % 2 )? ww : ww+1;
            hh = ( hh % 2 )? hh : hh+1;
            r[0] = cvRect( x1, y1, ww, hh );
            regions = r;
            ref_histos = compute_ref_histos( hsv_frame, regions, num_objects );
            particles = init_distribution( regions, ref_histos, num_objects, num_particles , frame->width, frame->height, U0);

            num = calculate_alive(particles,num_particles);
            if( show_all )
                for( j = num_particles - 1; j > 0; j-- )
                {
                    color = CV_RGB(255,255,0);
//                    display_particle( mRgb, particles[j], color );
                    x0 = round( particles[j].x - 0.5 * particles[j].s * particles[j].width );
                    y0 = round( particles[j].y - 0.5 * particles[j].s * particles[j].height );
                    x1 = x0 + round( particles[j].s * particles[j].width );
                    y1 = y0 + round( particles[j].s * particles[j].height );
                    xcenter1 = (x1 - x0)/2 -1;
                    ycenter1 = (y1 - y0)/2 -1;
                    rectangle( mRgb, Point( x0+xcenter1, y0+ycenter1), Point( x1-xcenter1, y1-ycenter1 ), color, 3, 8, 0 );
                }
            __android_log_print(ANDROID_LOG_VERBOSE, "show_all","mode  = %d",mode);
//            cvReleaseImage( &hsv_frame );
//            cvReleaseImage( &frame );
            __android_log_print(ANDROID_LOG_VERBOSE, "release","mode  = %d",mode);
            return MODE_TEST;
            break;
        case MODE_TEST:
            frame = &img;
            hsv_frame = bgr2hsv(frame);
            __android_log_print(ANDROID_LOG_VERBOSE, "mode","mode  = %d",mode);
            for( j = 0; j < num_particles; j++ )
            {
                particles[j] = transition( particles[j], frame->width, frame->height, U0,U1,regions, ref_histos);
                s = particles[j].s;
                if (particles[j].alive == 1)
                    particles[j].w = likelihood( hsv_frame, cvRound(particles[j].y),
                                                 cvRound( particles[j].x ),
                                                 cvRound( particles[j].width * s ),
                                                 cvRound( particles[j].height * s ),
                                                 particles[j].histo );
            }
            __android_log_print(ANDROID_LOG_VERBOSE, "after transition and likelihood","mode  = %d",mode);

            qsort( particles, num_particles, sizeof( particle ), &particle_cmp2 );

            normalize_weights( particles, num_particles );
            num = calculate_alive(particles,num_particles);
            new_particles = resample3(particles, num ,num_particles );

            particles = new_particles;

            if( show_all )
                for( j = num_particles - 1; j > 0; j-- )
                {
                    color = CV_RGB(255,255,0);
                    x0 = round( particles[j].x - 0.5 * particles[j].s * particles[j].width );
                    y0 = round( particles[j].y - 0.5 * particles[j].s * particles[j].height );
                    x1 = x0 + round( particles[j].s * particles[j].width );
                    y1 = y0 + round( particles[j].s * particles[j].height );
                    xcenter1 = (x1 - x0)/2 -1;
                    ycenter1 = (y1 - y0)/2 -1;
                    rectangle( mRgb, Point( x0+xcenter1, y0+ycenter1), Point( x1-xcenter1, y1-ycenter1 ), color, 3, 8, 0 );
                }
            __android_log_print(ANDROID_LOG_VERBOSE, "after show_all","mode  = %d",mode);

//            num = calculate_alive(particles,num_particles);
//
//            /* display all particles if requested */
//            qsort( particles, num_particles, sizeof( particle ), &particle_cmp );
////            if( show_all )
////                for( j = num_particles - 1; j > 0; j-- )
////                {
////                    color = CV_RGB(255,255,0);
//////                    display_particle( mRgb, particles[j], color );
////                    x0 = round( particles[j].x - 0.5 * particles[j].s * particles[j].width );
////                    y0 = round( particles[j].y - 0.5 * particles[j].s * particles[j].height );
////                    x1 = x0 + round( particles[j].s * particles[j].width );
////                    y1 = y0 + round( particles[j].s * particles[j].height );
////                    xcenter1 = (x1 - x0)/2 -1;
////                    ycenter1 = (y1 - y0)/2 -1;
////                    rectangle( mRgb, Point( x0+xcenter1, y0+ycenter1), Point( x1-xcenter1, y1-ycenter1 ), color, 1, 8, 0 );
////                }
////            __android_log_print(ANDROID_LOG_VERBOSE, "after show_all","mode  = %d",mode);
//
//            /* display most likely particle */
//            color = CV_RGB(255,0,0);
////            display_particle( mRgb, particles[0], color);
//            x0 = round( particles[0].x - 0.5 * particles[0].s * particles[0].width );
//            y0 = round( particles[0].y - 0.5 * particles[0].s * particles[0].height );
//            x1 = x0 + round( particles[0].s * particles[0].width );
//            y1 = y0 + round( particles[0].s * particles[0].height );
//            xcenter1 = (x1 - x0)/2 -1;
//            ycenter1 = (y1 - y0)/2 -1;
//            rectangle( mRgb, Point( x0+xcenter1, y0+ycenter1), Point( x1-xcenter1, y1-ycenter1 ), color, 1, 8, 0 );
//            __android_log_print(ANDROID_LOG_VERBOSE, "after showbest","mode  = %d",mode);
//            cvReleaseImage( &hsv_frame );
//            cvReleaseImage( &frame );
            return MODE_TEST;
            break;
    }

}

}
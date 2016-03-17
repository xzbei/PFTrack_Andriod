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

#define PARTICLES 500
#define MAX_OBJECTS 1
#define U0 0.20
#define U1 0.20

#define MODE_RESET 0
#define MODE_BEGIN 1
#define MODE_TRAIN 2
#define MODE_TEST 5
#define SCALE 3
#define THRES 0.05

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
//histogram** center_histos;

particle* particles, * new_particles;
CvScalar color;
CvRect* regions;
//CvRect* center_regions;


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
    frame = &img;

    switch (mode){
        case MODE_BEGIN:
            rectangle(mGr,Point((ycenter-ycenter/2)*SCALE,(xcenter - xcenter/2)*SCALE), Point((ycenter+ycenter/2)*SCALE,(xcenter+xcenter/2)*SCALE),Scalar(255,0,0,255),8);
//            __android_log_print(ANDROID_LOG_VERBOSE, "begin","rows  = %d",rows);
//            __android_log_print(ANDROID_LOG_VERBOSE, "begin","cols  = %d",cols);
//            __android_log_print(ANDROID_LOG_VERBOSE, "begin","frame_width  = %d",frame->width);
            return MODE_BEGIN;
            break;
        case MODE_RESET:
            return MODE_BEGIN;
            break;
        case MODE_TRAIN:
            hsv_frame = bgr2hsv(frame);
            CvRect* r;
            r = (CvRect*)malloc( 1 * sizeof( CvRect ) );
            xcenter = frame->width/2;
            ycenter = frame->height/2;

            x1 = round(xcenter - xcenter/2);
            y1 = round(ycenter - ycenter/2);

            ww = xcenter;
            hh = ycenter;

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
                    rectangle( mGr, Point( (x0+xcenter1)*SCALE, (y0+ycenter1)*SCALE), Point( (x1-xcenter1)*SCALE, (y1-ycenter1)*SCALE ), color, 3, 8, 0 );
                }
//            __android_log_print(ANDROID_LOG_VERBOSE, "show_all","mode  = %d",mode);
//            cvReleaseImage( &hsv_frame );
//            cvReleaseImage( &frame );
//            __android_log_print(ANDROID_LOG_VERBOSE, "release","mode  = %d",mode);
            return MODE_TEST;
            break;
        case MODE_TEST:

            hsv_frame = bgr2hsv(frame);
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

            qsort( particles, num_particles, sizeof( particle ), &particle_cmp2 );

            normalize_weights( particles, num_particles );
            num = calculate_alive(particles,num_particles);
            new_particles = resample3(particles, num ,num_particles );

            particles = new_particles;

            qsort( particles, num_particles, sizeof( particle ), &particle_cmp );

            if( show_all )
                for( j = 0; j < num_particles; j++ )
                {
                    if (particles[j].alive == 1)
                    {
                        color = CV_RGB(255, 255, 0);
                        x0 = round(particles[j].x - 0.5 * particles[j].s * particles[j].width);
                        y0 = round(particles[j].y - 0.5 * particles[j].s * particles[j].height);
                        x1 = x0 + round(particles[j].s * particles[j].width);
                        y1 = y0 + round(particles[j].s * particles[j].height);
                        xcenter1 = (x1 - x0) / 2 - 1;
                        ycenter1 = (y1 - y0) / 2 - 1;
                        rectangle(mGr, Point((x0 + xcenter1)*SCALE, (y0 + ycenter1)*SCALE),
                                  Point((x1 - xcenter1)*SCALE, (y1 - ycenter1)*SCALE), color, 3, 8, 0);
                    }
                }

            particle center_particle = Meanshift_cluster(particles,num,10,frame->width, frame->height);
            x0 = round( center_particle.x - 0.5 * center_particle.s * center_particle.width );
            y0 = round( center_particle.y - 0.5 * center_particle.s * center_particle.height );
            x1 = x0 + round( center_particle.s * center_particle.width );
            y1 = y0 + round( center_particle.s * center_particle.height );

            double score1 = likelihood(hsv_frame,cvRound(center_particle.y),cvRound(center_particle.x),cvRound(center_particle.width*center_particle.s),
                                       cvRound(center_particle.height*center_particle.s),center_particle.histo);
            __android_log_print(ANDROID_LOG_VERBOSE, "center_score","score1  = %f",score1);

            if (score1>=THRES)
                rectangle( mGr, Point( x0*SCALE, y0*SCALE), Point( x1*SCALE, y1*SCALE ), Scalar(255,0,0,255), 3, 8, 0 );
            else
                rectangle( mGr, Point( x0*SCALE, y0*SCALE), Point( x1*SCALE, y1*SCALE ), Scalar(255,255,0,255), 3, 8, 0 );

            return MODE_TEST;
            break;
    }

}

}
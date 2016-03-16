package org.opencv.samples.tutorial2;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.imgproc.Imgproc;
import org.bytedeco.javacpp.presets.gsl;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

public class Tutorial2Activity extends Activity implements CvCameraViewListener2 {
    private static final String    TAG = "OCVSample::Activity";

    private static final int       MODE_RESET     = 0;
    private static final int       MODE_BEGIN     = 1;
    private static final int       MODE_TRAIN    = 2;
    private static final int       MODE_TEST = 5;

    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;
    public  int                    Mode = 1;

//    private MenuItem               mItemPreviewRGBA;
//    private MenuItem               mItemPreviewGray;
//    private MenuItem               mItemPreviewCanny;
//    private MenuItem               mItemPreviewFeatures;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("mixed_sample");

                    mOpenCvCameraView.enableView();
                } break;

                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public Tutorial2Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial2_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial2_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        Mode = MODE_BEGIN;

        Button button1 = (Button) findViewById(R.id.button);//capture
        Button button2 = (Button) findViewById(R.id.button2);//reset
        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mode = MODE_TRAIN;
            }
        });
        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Mode = MODE_RESET;
            }
        });
    }

//    @Override
//    public boolean onCreateOptionsMenu(Menu menu) {
//        Log.i(TAG, "called onCreateOptionsMenu");
//        mItemPreviewRGBA = menu.add("Preview RGBA");
//        mItemPreviewGray = menu.add("Preview GRAY");
//        mItemPreviewCanny = menu.add("Canny");
//        mItemPreviewFeatures = menu.add("Find features");
//        return true;
//    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Mat mRgbaT = mRgba.t();
        Core.flip(mRgba.t(), mRgbaT, 1);
//        Imgproc.resize(mRgbaT, mRgbaT, mRgba.size());
        Imgproc.resize(mRgbaT, mRgbaT, mRgba.size(),0.01, 0.01, Imgproc.INTER_CUBIC);
        Mat mGrayT = mGray.t();
        Core.flip(mGray.t(), mGrayT, 1);
//        Imgproc.resize(mGrayT, mGrayT, mGray.size());
        Imgproc.resize(mGrayT, mGrayT, mGray.size(),0.01, 0.01, Imgproc.INTER_CUBIC);

//        Mat mRgbaTT;
        Imgproc.cvtColor(mRgbaT, mRgbaT, Imgproc.COLOR_RGBA2RGB);

        final int viewMode = Mode;
        Mode = FindFeatures(mGrayT.getNativeObjAddr(), mRgbaT.getNativeObjAddr(), viewMode);

        return mRgbaT;
    }
//
//    public void onClick()

//    public boolean onOptionsItemSelected(MenuItem item) {
//        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
//
//        if (item == mItemPreviewRGBA) {
//            mViewMode = VIEW_MODE_RGBA;
//        } else if (item == mItemPreviewGray) {
//            mViewMode = VIEW_MODE_GRAY;
//        } else if (item == mItemPreviewCanny) {
//            mViewMode = VIEW_MODE_CANNY;
//        } else if (item == mItemPreviewFeatures) {
//            mViewMode = VIEW_MODE_FEATURES;
//        }
//
//        return true;
//    }

    public native int FindFeatures(long matAddrGr, long matAddrRgba, int viewMode);
}

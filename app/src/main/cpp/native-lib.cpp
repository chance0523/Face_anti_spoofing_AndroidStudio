#include <jni.h>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>



#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace cv;
using namespace cv::ml;
#define hist4face 59*49
#define fe 58
Ptr<SVM> leftSVM = SVM::create();

/* lookup table for ULBP */
static int lookup[256] =
        {
                0,1,2,3,4,fe,5,6,7,fe,fe,fe,8,fe,9,10,11,fe,fe,fe,fe,fe,fe,fe,12,fe,fe,fe,13,fe,
                14,15,16,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,17,fe,fe,fe,fe,fe,fe,fe,18,
                fe,fe,fe,19,fe,20,21,22,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,
                fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,23,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,
                fe,fe,24,fe,fe,fe,fe,fe,fe,fe,25,fe,fe,fe,26,fe,27,28,29,30,fe,31,fe,fe,fe,32,fe,
                fe,fe,fe,fe,fe,fe,33,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,34,fe,fe,fe,fe,
                fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,fe,
                fe,35,36,37,fe,38,fe,fe,fe,39,fe,fe,fe,fe,fe,fe,fe,40,fe,fe,fe,fe,fe,fe,fe,fe,fe,
                fe,fe,fe,fe,fe,fe,41,42,43,fe,44,fe,fe,fe,45,fe,fe,fe,fe,fe,fe,fe,46,47,48,fe,49,
                fe,fe,fe,50,51,52,fe,53,54,55,56,57
        };


/* Uniform LBP */
Mat u_lbp(const Mat &frame)
{
    Mat result(frame.size(), CV_8UC1, Scalar(0));

    /* padding */
    for (int i = 0; i < frame.rows - 2; i++) {
        for (int j = 0; j < frame.cols - 2; j++) {
            int t[9];
            uchar center = t[1 * 3 + 1];
            uchar UniVal = 0;
            bool check[9];
            /* check clock wise */
            int calOrder[8] = { 1,2,5,8,7,6,3,0 };
            int change = 0;
            int k = 0;

            /* 3x3 box */
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    t[p * 3 + q] = frame.at<uchar>(i + p, j + q);
                }
            }
            //uchar currentValue = ReturnUniVal(t);

            /* 3x3 box */
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    if (q*p != 1)
                    {
                        /* compare pixel value */
                        check[p * 3 + q] = (center <= t[p * 3 + q]) ? true : false;
                    }
                }
            }

            /* consider change */
            for (k = 0; k < 8; k++) {
                if (check[calOrder[k]] != check[calOrder[k + 1]])
                {
                    change++;
                }
            }

            if (change <= 2) {
                for (k = 7; k >= 0; k--)
                    UniVal += check[calOrder[7 - k]] * (uchar)pow(2, k);
            }

            else
                UniVal = 5;

            //result.at<uchar>(i, j) = currentValue;
            result.at<uchar>(i + 1, j + 1) = UniVal;

        }
    }
    return result;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_chance_useopencvwithcmake_MainActivity_loadCascade(JNIEnv *env, jobject instance,
                                                            jstring cascadeFileName_) {

    const char *nativeFileNameString = env->GetStringUTFChars(cascadeFileName_, 0);


    string baseDir("/storage/emulated/0/");

    baseDir.append(nativeFileNameString);

    const char *pathDir = baseDir.c_str();

    jlong ret = 0;

    ret = (jlong) new CascadeClassifier(pathDir);

    if (((CascadeClassifier *) ret)->empty()) {

        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",
                            "CascadeClassifier로 로딩 실패  %s", nativeFileNameString);
    }
    else

        __android_log_print(ANDROID_LOG_DEBUG, "native-lib :: ",

                            "CascadeClassifier로 로딩 성공 %s", nativeFileNameString);

    env->ReleaseStringUTFChars(cascadeFileName_, nativeFileNameString);

    return ret;
}


extern "C"
JNIEXPORT jfloat JNICALL
Java_com_chance_useopencvwithcmake_MainActivity_detect(JNIEnv *env, jobject instance,
                                                       jlong cascadeClassifier_face,
                                                       jlong matAddrInput, jlong matAddrResult
                                                       //,jlong matAddrMerge
                                                       ) {

    Mat &img_input = *(Mat *) matAddrInput;

    Mat &img_result = *(Mat *) matAddrResult;

    //Mat &y_merge = *(Mat *) matAddrMerge;
    Mat y_merge;
    Ptr<SVM> my_svm = SVM::create();


    // my code
    //Mat frame;
    //Mat image;
    Mat frame_roi;
    Mat img_gray;
    //string inputName;
    vector<Rect> face;
    Mat image_resize;
    Mat frame_resize;
    Mat frame_ycbcr;
    Mat frame_hsv;
    Mat frame_split[3];
    Mat frame_split_hsv[3];
    vector<Mat>merge_mat(3);

    Mat frame_y_lbp_resize;

    float response;

    float denomi_y;
    float denomi_cr;
    float denomi_cb;
    float denomi_h;
    float denomi_s;
    float denomi_v;

    img_result = img_input.clone();
    cvtColor(img_input,img_gray,COLOR_BGR2GRAY);
    //Mat img_result2=img_result.clone();

    int x1=img_result.cols*0.7;
    int y1=100;
    int wi=320;
    int he=320;


    ((CascadeClassifier *) cascadeClassifier_face)->detectMultiScale(img_gray, face, 1.1, 2,
                                                                     0 | CASCADE_SCALE_IMAGE,
                                                                     Size(70, 70));

    if (face.size() != 0) {
        Point pt1(face[0].x, face[0].y);
        Point pt2(face[0].x + face[0].width, face[0].y + face[0].height);
        rectangle(img_result, pt1, pt2, Scalar(0, 255, 0), 3, 4, 0);

        frame_roi = img_input(Rect(face[0].x, face[0].y, face[0].width, face[0].height));
        //cvtColor(frame_roi,frame_roi,COLOR_BGR2GRAY);

        /* ROI */
        Rect rect(frame_roi.cols*0.12, frame_roi.rows*0.15, frame_roi.cols*0.73, frame_roi.rows*0.85);
        image_resize = frame_roi(rect);


        /* resize image */
        if (frame_roi.rows > 64 && frame_roi.cols > 64)
        {
            resize(image_resize, image_resize, Size(64, 64), 0, 0, INTER_AREA);
        }
        else
            resize(image_resize, image_resize, Size(64, 64), 0, 0, INTER_LINEAR);

        /*
        if (frame_roi.rows > 64 && frame_roi.cols > 64) {
            resize(frame_roi, frame_resize, Size(64, 64), 0, 0, INTER_AREA);
        } else
            resize(frame_roi, frame_resize, Size(64, 64), 0, 0, INTER_LINEAR);
        */

        cvtColor(image_resize, frame_ycbcr, COLOR_BGR2YCrCb);
        cvtColor(image_resize, frame_hsv, COLOR_BGR2HSV);

        split(frame_ycbcr, frame_split);
        split(frame_hsv, frame_split_hsv);

        Mat frame_y = frame_split[0];
        Mat frame_cr = frame_split[1];
        Mat frame_cb = frame_split[2];
        Mat frame_h = frame_split_hsv[0];
        Mat frame_s = frame_split_hsv[1];
        Mat frame_v = frame_split_hsv[2];


        /* uniform LBP */
        Mat frame_y_lbp = u_lbp(frame_y);
        Mat frame_cr_lbp = u_lbp(frame_cr);
        Mat frame_cb_lbp = u_lbp(frame_cb);
        Mat frame_h_lbp = u_lbp(frame_h);
        Mat frame_s_lbp = u_lbp(frame_s);
        Mat frame_v_lbp = u_lbp(frame_v);

        //////////////////////////////////////
        //           plot                  //
        ////////////////////////////////////
        resize(frame_y_lbp, frame_y_lbp_resize, Size(256, 256), 0, 0, INTER_AREA);

        for (int y=0;y<frame_y_lbp_resize.rows;y++){
            for (int x=0;x<frame_y_lbp_resize.cols;x++){
                uchar b=frame_y_lbp_resize.at<uchar>(y,x);

                img_result.at<Vec3b>(15+y,15+x)=b;
            }
        }



        /*
        float hist_y[59] = {0,};
        float hist_cr[59] = {0,};
        float hist_cb[59] = {0,};
        float hist_h[59] = {0,};
        float hist_s[59] = {0,};
        float hist_v[59] = {0,};
        */

        float face_feature_vector[6 * hist4face] = {0,};
        int x, y;

        /* make histogram (dimension = 59x49) */
        int Height = image_resize.rows;
        int Width = image_resize.cols;
        int bh = Height / 4;
        int bh_2 = bh / 2;
        int bw = Width / 4;
        int bw_2 = bw / 2;
        int i, j, k;
        int num = 0;
        float max = -1;

        /* padding (interval = Width/8 & Height/8) */
        for (y = 0; y <= Height - bh; y += bh_2) {
            for (x = 0; x <= Width - bw; x += bw_2) {

                float hist_y[59] = {0,};
                float hist_cr[59] = {0,};
                float hist_cb[59] = {0,};
                float hist_h[59] = {0,};
                float hist_s[59] = {0,};
                float hist_v[59] = {0,};

                denomi_y = 0;
                denomi_cr = 0;
                denomi_cb = 0;
                denomi_h = 0;
                denomi_s = 0;
                denomi_v = 0;

                /* read pixel value for small box */
                for (i = y; i < y + bh; i++) {
                    for (j = x; j < x + bw; j++) {
                        uchar p_y = frame_y_lbp.at<uchar>(i, j);
                        uchar p_cr = frame_cr_lbp.at<uchar>(i, j);
                        uchar p_cb = frame_cb_lbp.at<uchar>(i, j);
                        uchar p_h = frame_h_lbp.at<uchar>(i, j);
                        uchar p_s = frame_s_lbp.at<uchar>(i, j);
                        uchar p_v = frame_v_lbp.at<uchar>(i, j);

                        hist_y[lookup[p_y]]++;
                        hist_cr[lookup[p_cr]]++;
                        hist_cb[lookup[p_cb]]++;
                        hist_h[lookup[p_h]]++;
                        hist_s[lookup[p_s]]++;
                        hist_v[lookup[p_v]]++;
                    }
                }

                /* histogram normalization */
                for (k = 0; k < 59; k++) {
                    denomi_y += hist_y[k] * hist_y[k];
                    denomi_cr += hist_cr[k] * hist_cr[k];
                    denomi_cb += hist_cb[k] * hist_cb[k];
                    denomi_h += hist_h[k] * hist_h[k];
                    denomi_s += hist_s[k] * hist_s[k];
                    denomi_v += hist_v[k] * hist_v[k];
                }

                denomi_y = sqrt(denomi_y);
                denomi_cr = sqrt(denomi_cr);
                denomi_cb = sqrt(denomi_cb);
                denomi_h = sqrt(denomi_h);
                denomi_s = sqrt(denomi_s);
                denomi_v = sqrt(denomi_v);

                if (denomi_y == 0)hist_y[59] = {0,};
                if (denomi_cr == 0)hist_cr[59] = {0,};
                if (denomi_cb == 0)hist_cb[59] = {0,};
                if (denomi_h == 0)hist_h[59] = {0,};
                if (denomi_s == 0)hist_s[59] = {0,};
                if (denomi_v == 0)hist_v[59] = {0,};

                for (k = 0; k < 59; k++) {
                    hist_y[k] /= denomi_y;
                    hist_cr[k] /= denomi_cr;
                    hist_cb[k] /= denomi_cb;
                    hist_h[k] /= denomi_h;
                    hist_s[k] /= denomi_s;
                    hist_v[k] /= denomi_v;

                    /* store at feature vector */
                    face_feature_vector[num * 59 + k] = hist_y[k];
                    face_feature_vector[hist4face + num * 59 + k] = hist_cr[k];
                    face_feature_vector[2 * hist4face + num * 59 + k] = hist_cb[k];
                    face_feature_vector[3 * hist4face + num * 59 + k] = hist_h[k];
                    face_feature_vector[4 * hist4face + num * 59 + k] = hist_s[k];
                    face_feature_vector[5 * hist4face + num * 59 + k] = hist_v[k];
                }
                num++;
            }
        }

        Mat testData(1, 6 * hist4face, CV_32FC1, face_feature_vector);
       float response2 = leftSVM->predict(testData, noArray(), StatModel::RAW_OUTPUT);


       response = response2;

        //img_result&=frame_y_lbp;
    }
    else response = 0;


    return response;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_chance_useopencvwithcmake_MainActivity_CloadSVM(JNIEnv *env, jobject instance,
                                                            jstring leftSVMPath_) {
    const char *nativeFileNameString = env->GetStringUTFChars(leftSVMPath_, 0);


    string baseDir("/storage/emulated/0/");

    baseDir.append(nativeFileNameString);

    const char *pathDir = baseDir.c_str();


    leftSVM = SVM::load(pathDir);


    env->ReleaseStringUTFChars(leftSVMPath_, nativeFileNameString);

    //return ret;
}
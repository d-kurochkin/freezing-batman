//Qt include section
#include <QDebug>

//OpenCV include section
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

//захватываемый кадр
Mat frame, src;
vector<Mat> channels;
RNG rng(12345);

int thresh = 100;
const int mask = 9;
int erosion_size = 1;
int approx_size = 4;
int approx_error = 0;
int minimal_area = 100;
int match_value = 10;


vector<vector<Point> > contours;
vector<Vec4i> hierarchy;


void sobelDerivatives(const Mat &input, Mat &result);

int main()
{

    CvCapture* capture = cvCreateCameraCapture(1); //cvCaptureFromCAM( 0 );
    assert(capture);

    /// Logitech Quickcam Sphere AF
//        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1600);
//        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 1200);


    // узнаем ширину и высоту кадра
    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

    qDebug("[i] %.0f x %.0f\n", width, height);


    cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);

    qDebug("[i] press Enter for capture image and Esc for quit!\n\n");

    /// Переменные для сохранения файлов
    int counter=0;
    char filename[512];


//    createTrackbar("Erosion:", "result", &erosion_size, 255);
    createTrackbar("", "result", &approx_size, 20);
    waitKey(1000);

    while(true){
        ///начало отсчета времени
        double t = (double)getTickCount();

        /// получаем кадр
        frame = cvQueryFrame(capture);

        /// используем существующее изображение
        //frame = imread("C:\\Development\\Computersehen\\Samples\\GOPR0948.JPG");
        //resize(frame, frame, Size(), 0.4, 0.4, INTER_CUBIC);

        src = frame.clone();

        ///преобразуем в оттенки серого
        cvtColor(frame, frame, CV_RGB2GRAY);

        GaussianBlur(frame, frame, Size( 5, 5 ), 0, 0);


        /// Detect edges using canny
        Canny(frame, frame, thresh, thresh*2, 3 );

        /// Use adaptive threshold
        //adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 2);
        //medianBlur(frame, frame, 5);

        // Show preprocessed image
        imshow("capture", frame);

        /// Find contours
        findContours(frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Draw contours
        //Mat drawing = Mat::zeros( frame.size(), CV_8UC3 );
        Mat drawing = src.clone();

        int contours_count = 0;
        for( int i = 0; i< contours.size(); i++ )
        {

            double length = arcLength(contours[i], true);
            vector<Point> approx;
            approxPolyDP(contours[i], approx, 0.02*length, true);
//                contours[i] = approx;

            if (contourArea(approx) > minimal_area && isContourConvex(approx) && approx.size() == approx_size) {
                contours_count += 1;
//                if (contours[i].size() == approx_size && contourArea(contours[i]) > 100 ) {


//                    double I1 = matchShapes(contours[i], crossTemplate, CV_CONTOURS_MATCH_I1, 0);
//                    double I2 = matchShapes(contours[i], crossTemplate, CV_CONTOURS_MATCH_I2, 0);
//                    double I3 = matchShapes(contours[i], crossTemplate, CV_CONTOURS_MATCH_I3, 0);

//                    if (I1 < match_value/1.0) {
//                        qDebug() << "I1 = " << I1 << "\tI2 = " << I2 << "\tI3 = " << I3;

                Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );

                /// Draw center of contour
                Point2f center;
                float radius;
                minEnclosingCircle(approx, center, radius );
                circle( drawing, center, 1, color);

//                    }
                }
        }

        qDebug() << "Contours count " << contours_count;

        /// Calculate time
        t = ((double)getTickCount() - t)/getTickFrequency();
        qDebug() << "Times passed in seconds: " << t;


        /// Show in a window
        imshow( "result", drawing );




        // ожидаем нажатия клавишь
        char c = waitKey(33);
        if (c == 27) { // нажата ESC
                break;
        }
        else if(c == 13) { // Enter
                // сохраняем кадр в файл
                sprintf(filename, "Image%d.jpg", counter);
                qDebug("[i] capture... %s\n", filename);
                imwrite(filename, frame);
                counter++;
        }
    }
    // освобождаем ресурсы
    cvReleaseCapture( &capture );
//        cvDestroyWindow("adaptive_threshold_mean");
    return 0;
}

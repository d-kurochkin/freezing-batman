#include <QDebug>

//OpenCV include section
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "shape.h"

using namespace cv;
using namespace std;

//захватываемый кадр
Mat frame, src;
vector<Mat> channels;
RNG rng(12345);

/// Массив с фигурами
vector<Shape> shapes;

/// Параметры препроцессинга
int thresh = 100;

/// Переменные для сохранения файлов
int counter=0;
char filename[512];


void preprocessImage(Mat &frame);
void processContours(Mat &frame);
void drawContours();

int main()
{

    CvCapture* capture = cvCreateCameraCapture(1); //cvCaptureFromCAM( 0 );
    assert(capture);

    /// Logitech Quickcam Sphere AF
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280 );
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 960 );


    // узнаем ширину и высоту кадра
    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    qDebug("[i] %.0f x %.0f\n", width, height);

    cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
    //resizeWindow("capture", 1024, 768);
    //resizeWindow("result", 1024, 768);

    qDebug("[i] press Enter for capture image and Esc for quit!\n\n");


    createTrackbar("Threshold:", "result", &thresh, 255);
    //createTrackbar("", "result", &approx_size, 20);
    waitKey(1000);

    /// Основной цикл программы
    while(true){
        ///начало отсчета времени
        double t = (double)getTickCount();

        /// получаем кадр
        frame = cvQueryFrame(capture);

        src = frame.clone();

        preprocessImage(frame);

        processContours(frame);

        drawContours();

        /// Calculate time
        t = ((double)getTickCount() - t)/getTickFrequency();
        qDebug() << "Times passed in seconds: " << t;

        // ожидаем нажатия клавиш
        char c = waitKey(33);
        if (c == 27) {
            // нажата ESC
            break;
        }
        else if(c == 13) { // Enter
            /*@todo Добавить сохранение развернутых файлов (оригинал, обработанное, результат контурного анализа)*/
            // сохраняем кадр в файл
            sprintf(filename, "Image%d.jpg", counter);
            qDebug("[i] capture... %s\n", filename);
            imwrite(filename, frame);
            counter++;
        }
    }
    // освобождаем ресурсы
    cvReleaseCapture( &capture );
//  cvDestroyWindow("adaptive_threshold_mean");
    return 0;
}


void preprocessImage(Mat &frame) {

    ///преобразуем в оттенки серого
    cvtColor(frame, frame, CV_RGB2GRAY);

    equalizeHist(frame, frame);

    GaussianBlur(frame, frame, Size( 5, 5 ), 0, 0);


    /// Detect edges using canny
    Canny(frame, frame, thresh, thresh*2, 3);

    /// Use adaptive threshold
    //adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 7, 2);
    //medianBlur(frame, frame, 5);

    // Show preprocessed image
    imshow("capture", frame);
}

void processContours(Mat &frame)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Find contours
    findContours(frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Очищаем вектор с формами
    shapes.clear();

    /// Fill shapes vector
    for( int i = 0; i< contours.size(); i++ )
    {
        /// Approximate contour to find border count
        int shapeType = SHAPE_NONE;
        shapeType = Shape::classifyShape(contours[i]);
        //contours[i] = approx;

        if (shapeType != SHAPE_NONE) {

            if (shapes.size() == 0) {
                Shape temp(contours[i]);
                shapes.push_back(temp);
            } else {
                bool isAdded = false;

                ///проверяем относится ли контур к какой-либо фигуре, если да, то поглащаем меньшую фигуру большей
                for(int j=0; j<shapes.size(); ++j) {
                    if (shapes[j].shapeType == shapeType && shapes[j].centerIsInside(contours[i])) {
                        shapes[j].mergeContours(contours[i]);
                        isAdded = true;
                        break;
                    }
                }

                if (!isAdded) {
                    Shape temp(contours[i]);
                    shapes.push_back(temp);
                }
            }
        }
    }
}

void drawContours() {
    ///Draw shapes
    Mat drawing = src.clone();
    vector<vector<Point> > resultContours;

    for (int i=0; i<shapes.size(); ++i) {
        resultContours.push_back(shapes[i].shapeContour);

        /// Draw contour
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, resultContours, i, color, 2, 8, NULL, 0, Point() );

        /// Draw center of contour
        Point2f center;
        float radius;
        minEnclosingCircle(resultContours[i], center, radius );
        circle(drawing, center, 3, color, 2);
    }

    /// Show in a window
    imshow( "result", drawing );
}

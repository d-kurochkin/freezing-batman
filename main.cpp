#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "shape.h"

using namespace cv;
using namespace std;

//захватываемый кадр
Mat frame, src;

/// Массив с фигурами
vector<Shape> shapes;
vector<Shape> triangles;
vector<Shape> squares;
vector<Shape> hexagons;
vector<Shape> circles;

// Распознанные фигуры
Shape platformShape;
Shape centerShape;

/// Параметры препроцессинга
int thresh = 50;
int color_coeff = 70;
int min_eucl_dist = 50;

/// Переменные для сохранения файлов
int counter=0;
char filename[512];

void initInterface();
void preprocessImage(Mat &frame);
void processContours(Mat &frame);
void processShapes();
void drawContours();

int main()
{
    CvCapture* capture = cvCreateCameraCapture(0); //cvCaptureFromCAM( 0 );
    assert(capture);

    /// Logitech Quickcam Sphere AF
//    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 1280 );
//    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 960 );

    // узнаем ширину и высоту кадра
    double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    qDebug("[i] %.0f x %.0f\n", width, height);

    initInterface();

    /// Основной цикл программы
    while(true){
        /// получаем кадр
        frame = cvQueryFrame(capture);

        ///начало отсчета времени
        double t = (double)getTickCount();

        src = frame.clone();

        preprocessImage(frame);
        processContours(frame);
        processShapes();
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

    return 0;
}

void initInterface() {
    cvNamedWindow("settings", CV_WINDOW_NORMAL);
    cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
    //resizeWindow("capture", 1024, 768);
    //resizeWindow("result", 1024, 768);

    createTrackbar("Threshold:", "settings", &thresh, 255);
    createTrackbar("Color coeff", "settings", &color_coeff, 255);
    createTrackbar("Minimal euclidian distance", "settings", &min_eucl_dist, 150);
    waitKey(1000);

}

void preprocessImage(Mat &frame) {

    Mat hsv, yuv;
    vector<Mat> channels_hsv, channels_yuv;

    ///преобразуем в оттенки серого
    cvtColor(frame, hsv, CV_RGB2HSV);
    cvtColor(frame, yuv, CV_RGB2YUV);
    split(hsv, channels_hsv);
    split(yuv, channels_yuv);

    Mat temp;
    addWeighted(channels_hsv[1], 0.5, channels_hsv[2], 0.5, 0, temp);
    multiply(temp, channels_yuv[0], frame, 1.0/color_coeff);
    //imshow("convert_color", frame);

    /// Выравнивание гистограммы
    //equalizeHist(frame, frame);
    /// Сглаживание изображения
    GaussianBlur(frame, frame, Size( 5, 5 ), 0, 0);
    /// Detect edges using canny
    Canny(frame, frame, thresh, thresh*2, 3);

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


        if (shapeType != SHAPE_NONE) {

            if (shapes.size() == 0) {
                Shape temp(contours[i]);
                shapes.push_back(temp);
            } else {
                bool isAdded = false;

                ///проверяем относится ли контур к какой-либо фигуре, если да, то поглощаем меньшую фигуру большей
                for(int j=0; j<shapes.size(); ++j) {
                    double eucliadianDistance = 0;
                    bool isInside = shapes[j].centerIsInside(contours[i], eucliadianDistance);

                    if (shapes[j].shapeType == shapeType && isInside && eucliadianDistance < min_eucl_dist) {
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

void pushShape_1(vector<Shape> &items) {
    for(auto item : items) {
        double eucliadianDistance = 0;
        bool centerIsInside = platformShape.centerIsInside(item.shapeContour, eucliadianDistance);
        double area_ratio = centerShape.shapeArea / item.shapeArea;

        if (centerIsInside && eucliadianDistance > min_eucl_dist && area_ratio < 2 && area_ratio > 0.5) {
            shapes.push_back(item);
            break;
        }
    }
}

void processShapes() {
    //Очистка
    triangles.clear();
    squares.clear();
    hexagons.clear();
    circles.clear();

    centerShape = Shape();
    platformShape = Shape();


    //Помещаем фигуры в соответствующие массивы
    for (int i = 0; i < shapes.size(); ++i) {
        switch (shapes[i].shapeType) {
        case SHAPE_TRIANGLE:
            triangles.push_back(shapes[i]);
            break;
        case SHAPE_SQUARE:
            squares.push_back(shapes[i]);
            break;
        case SHAPE_HEXAGON:
            hexagons.push_back(shapes[i]);
            break;
        case SHAPE_CIRCLE:
            circles.push_back(shapes[i]);
            break;
        default:
            break;
        }
    }

    qDebug() << shapes.size() << "\t" << triangles.size() << "\t" << squares.size() << "\t" << hexagons.size() << "\t" << circles.size();
    shapes.clear();

    //Выполняем поиск платформы по наивному алгоритму


    for (int circle_item = 0; circle_item < circles.size(); ++circle_item) {
        for (int square_item = 0; square_item < squares.size(); ++square_item) {
            double eucliadianDistance = 0;
            bool centerIsInside = circles[circle_item].centerIsInside(squares[square_item].shapeContour, eucliadianDistance);
            double area_ratio = squares[square_item].shapeArea / circles[circle_item].shapeArea;

            if (centerIsInside && eucliadianDistance < min_eucl_dist && area_ratio > 2.5) {
                if (squares[square_item].shapeArea > platformShape.shapeArea && circles[circle_item].shapeArea > centerShape.shapeArea) {
                    platformShape = squares[square_item];
                    platformShape.shapeType = SHAPE_PLATFORM;

                    centerShape = circles[circle_item];
                    centerShape.shapeType = SHAPE_CENTER;
                }
            }
        }
    }

    //Добавляем найденные центр и платформу в массив фигур
    if (platformShape.shapeArea > 0 && centerShape.shapeArea > 0) {
        shapes.push_back(platformShape);
        shapes.push_back(centerShape);

        pushShape_1(triangles);
        pushShape_1(squares);
        pushShape_1(hexagons);
        pushShape_1(circles);
    }

}

void drawContours() {
    ///Draw shapes
    Mat drawing = src.clone();
    vector<vector<Point> > resultContours;

    for (int i=0; i<shapes.size(); ++i) {
        resultContours.push_back(shapes[i].shapeContour);

        /// Draw contour

        drawContours( drawing, resultContours, i,  SHAPE_COLORS[shapes[i].shapeType], 2, 8, NULL, 0, Point() );

        /// Draw center of contour
        Point2f center;
        float radius;
        minEnclosingCircle(resultContours[i], center, radius );
        circle(drawing, center, 3,  SHAPE_COLORS[shapes[i].shapeType], 2);


        //дорисовать минимальный описывающий прямоугольник и окружности
//        Scalar color = Scalar(255, 255, 255);
//        RotatedRect minRect = minAreaRect(Mat(resultContours[i]));
//        Point2f rect_points[4]; minRect.points( rect_points );


//        Size2f size =minRect.size;
//        float max = size.height > size.width ? size.height : size.width;

//        radius = max / 0.46;
//        circle(drawing, shapes[i].shapeCenter, radius, color, 2, 8 );
    }

    /// Show in a window
    imshow( "result", drawing );
}

#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "shape.h"

using namespace cv;
using namespace std;

/// захватываемый кадр
Mat frame, gray_src, src, drawing;

/// Массив с фигурами
vector<Shape> shapes;
vector<Shape> triangles;
vector<Shape> squares;
vector<Shape> hexagons;
vector<Shape> circles;

/// Распознанные фигуры
Shape platformShape;
Shape centerShape;
Shape triangleShape;
Shape squareShape;
Shape hexagonShape;
Shape circleShape;

/// Параметры препроцессинга
int thresh = 50;
int color_coeff = 70;
int min_eucl_dist = 50;
int center_detect_threshold = 100;
double center_detect_radius = 1.3;
int raduis_coef = 375;
int class_threshold = 80;


/// Переменные для сохранения файлов
int counter=0;
char filename[512];

void initInterface();
void preprocessImage(Mat &frame);
void processContours(Mat &frame);
void processShapes();
void processEdgeShapes();
void drawShapes();

///Platform detection methods
bool simpleDetection();
bool centerFirstDetection();


//Detect altitude
double frameHeight;
double frameWidth;
float calculateAltitude();
void calculateOffset();
void calculatePlatformAngle();

int main()
{
    CvCapture* capture = cvCreateCameraCapture(0); //cvCaptureFromCAM( 0 );
    assert(capture);

    /// Параметры необходимые для платы видеозахвата
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 720);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);
    //cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, 30);

    // узнаем ширину и высоту кадра
    frameWidth = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    frameHeight = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    //qDebug("[i] %.0f x %.0f\n", frameWidth, frameHeight);


    initInterface();
    waitKey(1000);


    /// Основной цикл программы
    while(true){
        /// получаем кадр
        frame = cvQueryFrame(capture);

        ///начало отсчета времени
        //double t = (double)getTickCount();

        preprocessImage(frame);
        processContours(frame);
        processShapes();
        drawShapes();



        /// Calculate time
        //t = ((double)getTickCount() - t)/getTickFrequency();
        //qDebug() << "Times passed in seconds: " << t;

        // ожидаем нажатия клавиш
        char c = waitKey(33);
        if (c == 27) {
            // нажата ESC
            break;
        }
        else if(c == 13) { // Enter
            // сохраняем кадр в файл
            sprintf(filename, "Image%d.jpg", counter);
            //qDebug("[i] capture... %s\n", filename);
            imwrite(filename, drawing);
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
    //resizeWindow("settings", 1024, 768);
    //resizeWindow("result", 1024, 768);

    createTrackbar("THR:", "settings", &thresh, 255);
    createTrackbar("CCOEF", "settings", &color_coeff, 255);
    createTrackbar("MED", "settings", &min_eucl_dist, 150);
    createTrackbar("CDT", "settings", &center_detect_threshold, 150);
    createTrackbar("RCOEF", "settings", &raduis_coef, 1000);
    createTrackbar("CLSTHR", "settings", &class_threshold, 150);
    //createTrackbar("DCOEF/10", "settings", &distance_coeff, 500);

    waitKey(1000);
}

void preprocessImage(Mat &frame) {

    Mat hsv, yuv;
    vector<Mat> channels_hsv, channels_yuv;
    src = frame.clone();
    drawing = src.clone();

    ///преобразуем в оттенки серого
    cvtColor(frame, hsv, CV_RGB2HSV);
    cvtColor(frame, yuv, CV_RGB2YUV);
    split(hsv, channels_hsv);
    split(yuv, channels_yuv);

    Mat temp;
    addWeighted(channels_hsv[1], 0.5, channels_hsv[2], 0.5, 0, temp);
    multiply(temp, channels_yuv[0], frame, 1.0/color_coeff);

    imshow("grayscale", frame);
    //equalizeHist(frame, frame);

    GaussianBlur(frame, frame, Size( 5, 5 ), 0, 0);
    gray_src = frame.clone();

    /// Detect edges using canny
    Canny(frame, frame, thresh, thresh*2, 3);
    imshow("Canny", frame);
}


void processContours(Mat &frame)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Find contours
    findContours(frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// Очищаем вектор с формами
    shapes.clear();

    /// Fill shapes vector
    for( int i = 0; i< contours.size(); i++ )
    {
        /// Approximate contour to find border count
        int shapeType = SHAPE_NONE;
        shapeType = Shape::classifyShape(contours[i], (double)(class_threshold/100.0));


        if (shapeType != SHAPE_NONE) {

            if (shapes.size() == 0) {
                Shape temp(contours[i], shapeType);
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
                    Shape temp(contours[i], shapeType);
                    shapes.push_back(temp);
                }
            }
        }
    }
}

void processShapes() {
    /// Clear formes arraysS
    triangles.clear();
    squares.clear();
    hexagons.clear();
    circles.clear();

    /// Put shapes into formes arrays
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

    /// Clear shapes base arrays
    shapes.clear();

    /// Detect platform with simple method
    if (simpleDetection()) {
        //qDebug() << "Simple method";
    } else if (centerFirstDetection()) {
        //qDebug() << "Center first method";
    } else {
        //qDebug() << "No platform";
    }

    processEdgeShapes();

    calculateAltitude();
    calculateOffset();
    calculatePlatformAngle();
}

void drawShapes() {
    shapes.clear();
    if (centerShape.shapeArea > 0) {
        shapes.push_back(centerShape);

        if (triangleShape.shapeArea > 0)
            shapes.push_back(triangleShape);

        if (squareShape.shapeArea > 0)
            shapes.push_back(squareShape);

        if (hexagonShape.shapeArea > 0)
            shapes.push_back(hexagonShape);

        if (circleShape.shapeArea > 0)
            shapes.push_back(circleShape);

        if (platformShape.shapeArea > 0)
            shapes.push_back(platformShape);
    }

    ///Draw shapes
    vector<vector<Point> > resultContours;

    for (int i=0; i<shapes.size(); ++i) {
        resultContours.push_back(shapes[i].shapeContour);

        /// Draw contour
        drawContours(drawing, resultContours, i,  SHAPE_COLORS[shapes[i].shapeType], 2, 8, NULL, 0, Point() );

        /// Draw center of shape
        Point2f center;
        float radius;
        minEnclosingCircle(resultContours[i], center, radius );
        circle(drawing, center, 3,  SHAPE_COLORS[shapes[i].shapeType], 2);
    }

    if (centerShape.shapeArea > 0) {
        QString text;

        putText(drawing, "Center of platform", Point(1, 10), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1,8);

        text = QString("x = %1").arg((int) centerShape.shapeCenter.x);
        putText(drawing, text.toStdString(), Point(1, 25), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);

        text = QString("y = %1").arg((int) centerShape.shapeCenter.y);
        putText(drawing, text.toStdString(), Point(1, 40), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);
    }

    /// Show in a window
    imshow( "result", drawing );
}

void pushShape_1(vector<Shape> &items) {
    for(Shape item : items) {
        double eucliadianDistance = 0;
        bool centerIsInside = platformShape.centerIsInside(item.shapeContour, eucliadianDistance);
        double area_ratio = centerShape.shapeArea / item.shapeArea;

        if (centerIsInside && eucliadianDistance > min_eucl_dist && area_ratio < 2 && area_ratio > 0.5) {
            shapes.push_back(item);
            break;
        }
    }
}

bool simpleDetection() {
    centerShape = Shape();
    platformShape = Shape();
    triangleShape = Shape();
    squareShape = Shape();
    hexagonShape = Shape();
    circleShape = Shape();

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

        return true;
    } else {
        return false;
    }
}

void pushShape_2(vector<Shape> &items, Shape &center) {
    for(Shape item : items) {
        double eucliadianDistance = 0;

        Point diff = item.shapeCenter - center.shapeCenter;
        eucliadianDistance = sqrt(diff.x*diff.x + diff.y*diff.y);

        if (eucliadianDistance <= center.shapeRadius*raduis_coef/100.0 && eucliadianDistance > 10) {
            shapes.push_back(item);
        }
    }
}

bool centerFirstDetection() {
    centerShape = Shape();
    platformShape = Shape();

    int centerCrossingCount = 32000;
    for(Shape item : circles) {
        int count = Shape::detectCentralShape(gray_src, item.shapeCenter, item.shapeRadius*center_detect_radius, center_detect_threshold);

        if (count < centerCrossingCount /*&& item.shapeArea > centerShape.shapeArea*/) {
            centerShape = item;
            centerCrossingCount = count;
        }
    }

    if (centerShape.shapeArea > 0) {
        centerShape.shapeType = SHAPE_CENTER;
        shapes.push_back(centerShape);

        double radius = centerShape.shapeRadius*raduis_coef/100.0;
        circle(src, centerShape.shapeCenter, radius,  cv::Scalar(96, 115, 27), 1);

        pushShape_2(triangles, centerShape);
        pushShape_2(squares, centerShape);
        pushShape_2(hexagons, centerShape);
        pushShape_2(circles, centerShape);

        return true;
    } else {
        return false;
    }
}

void processEdgeShapes() {
    double eucliadianDistance = 0;
    bool is_inside = false;

    for(Shape item : shapes) {
        switch (item.shapeType) {
        case SHAPE_TRIANGLE:

            if (circleShape.shapeArea > 0) {
                is_inside = circleShape.centerIsInside(item.shapeContour, eucliadianDistance);
            } else {
                is_inside =  false;
            }
            if (item.shapeArea > triangleShape.shapeArea && !is_inside) {
                triangleShape = item;
            }
            break;
        case SHAPE_SQUARE:
            if (item.shapeArea > squareShape.shapeArea) {
                squareShape = item;
            }
            break;
        case SHAPE_HEXAGON:
            if (item.shapeArea > hexagonShape.shapeArea) {
                hexagonShape = item;
            }
            break;
        case SHAPE_CIRCLE:
            if (hexagonShape.shapeArea > 0) {
                is_inside = hexagonShape.centerIsInside(item.shapeContour, eucliadianDistance);
            } else {
                is_inside =  false;
            }
            if (item.shapeArea > circleShape.shapeArea && !is_inside) {
                circleShape = item;
            }
            break;
        }
    }
}

float calculateAltitude() {
    if (centerShape.shapeArea > 0) {
        double frameFactor = qSqrt(frameHeight*frameWidth);
        double shapeFactor = qSqrt(centerShape.shapeArea);

        double x = frameFactor / shapeFactor;
        double distance = 0.0064*x*x + 38.112*x + 5;
        //double distance = x;

        QString text = QString("Distance = %1").arg(distance);
        putText(drawing, text.toStdString(), Point(1, 55), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);

        text = QString("Relative distance = %1").arg(x);
        putText(drawing, text.toStdString(), Point(1, 70), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);
    }

    return 0;
}


void calculateOffset() {
    if (centerShape.shapeArea > 0) {
        double x = (1 - (2 * centerShape.shapeCenter.x / frameWidth))*100;
        double y = (1 - (2 * centerShape.shapeCenter.y / frameHeight))*100;

        QString text = QString("X offset = %1% Y offset = %2%").arg(QString::number((int)x), QString::number((int)y));
        putText(drawing, text.toStdString(), Point(1, 85), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);
    }
}


double calculateAngle(Shape &item) {
    double x = item.shapeCenter.x - centerShape.shapeCenter.x;
    double y = item.shapeCenter.y - centerShape.shapeCenter.y;
    double angle = 0;


    if (x == 0) {
        if (y < 0) {
            angle = 180;
        } else {
            angle = 0;
        }
    }
    else {
        angle = qAtan(y/x)*180.0/PI;
        if (x > 0) {
            angle += 90;
        } else {
            angle += 270;
        }

    }
    return angle;
}

void calculatePlatformAngle() {
    if (centerShape.shapeArea > 0) {
        vector<int> angles;
        int angle;

        if (triangleShape.shapeArea > 0) {
            angle =(int)(calculateAngle(triangleShape) + (360 - 219)) % 360;
            angles.push_back(angle);
        }

        if (squareShape.shapeArea > 0) {
            angle =(int)(calculateAngle(squareShape) + (360 - 45)) % 360;
            angles.push_back(angle);
        }

        if (hexagonShape.shapeArea > 0) {
            angle =(int)(calculateAngle(hexagonShape) + (360 - 315)) % 360;
            angles.push_back(angle);
        }

        if (circleShape.shapeArea > 0) {
            angle =(int)(calculateAngle(circleShape) + (360 - 135)) % 360;
            angles.push_back(angle);
        }

        if (angles.size() > 0) {

            int average = 0;

            int min = *min_element(angles.begin(), angles.end());
            int max = *max_element(angles.begin(), angles.end());
            if ((max - min) > 300) {
                for(auto item : angles) {
                    if ((item + 300) < max) {
                        average += item + 360;
                    } else {
                        average += item;
                    }
                }
                average /= angles.size();
            } else {
                average = accumulate(angles.begin(), angles.end(), 0) / angles.size();
            }

            if(average > 180) {
                average = -1 * (360 - average);
            }

            QString text = QString("Platform angle = %1").arg(QString::number(average));
            putText(drawing, text.toStdString(), Point(1, 100), FONT_HERSHEY_PLAIN, 1, Scalar(28, 232, 0), 1, 8);
        }

    }
}




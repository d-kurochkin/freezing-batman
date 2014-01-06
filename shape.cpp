#include "shape.h"



Shape::Shape(std::vector<cv::Point> &contour)
{
    shapeContour = contour;

    float radius;
    cv::minEnclosingCircle(shapeContour, shapeCenter, radius);

    shapeArea = cv::contourArea(shapeContour);

    shapeChildrenCount += 1;
    qDebug("Crate first element");
}

bool Shape::centerIsInside(std::vector<cv::Point> &contour)
{
    bool fl = false;
    cv::Point2f center;
    float radius;

    cv::minEnclosingCircle(contour, center, radius);

    double distance;
    distance = cv::pointPolygonTest(shapeContour, center, false);

    if (distance > 0) {
        fl = true;
    }

    return fl;
}

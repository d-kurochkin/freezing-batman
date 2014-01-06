#include "shape.h"



Shape::Shape(std::vector<cv::Point> &contour)
{
    shapeContour = contour;

    float radius;
    cv::minEnclosingCircle(shapeContour, shapeCenter, radius);

    shapeArea = cv::contourArea(shapeContour);

    shapeChildrenCount += 1;
}

bool Shape::centerIsInside(std::vector<cv::Point> &contour)
{
    bool fl = false;

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);

    double distance;
    distance = cv::pointPolygonTest(shapeContour, center, false);

    double eucliadianDistance = 0;
    cv::Point diff = shapeCenter - center;
    eucliadianDistance = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
    qDebug() << "eucliadianDistance = " << eucliadianDistance;

    if (distance >= 0 && eucliadianDistance < 50) {
        fl = true;
    }

    return fl;
}

void Shape::mergeContours(std::vector<cv::Point> &contour)
{
    float area = cv::contourArea(contour);

    if (area > shapeArea) {
        /// Set new area of shape
        shapeArea = area;

        /// Calculate new center of shape
        float radius;
        cv::minEnclosingCircle(contour, shapeCenter, radius);

        shapeContour = contour;
    }

    shapeChildrenCount += 1;
}

void Shape::approximateContour(std::vector<cv::Point> &contour, double &area, int &sides, bool &isClosed)
{
    double length = arcLength(contour, true);
    double epsilon = 0.02*length;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, true);

    area = contourArea(approx);

    isClosed = isContourConvex(approx);

    sides = approx.size();
}

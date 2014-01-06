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

    if (distance >= 0) {
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

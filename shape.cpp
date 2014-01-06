#include "shape.h"

Shape::Shape()
{
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

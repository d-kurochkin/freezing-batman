#include "shape.h"

double angle_point (point a, point b, point c)
{
   double x1 = a.x - b.x, x2 = c.x - b.x;
   double y1 = a.y - b.y, y2 = c.y - b.y;
   double d1 = sqrt (x1 * x1 + y1 * y1);
   double d2 = sqrt (x2 * x2 + y2 * y2);
   return acos ((x1 * x2 + y1 * y2) / (d1 * d2));
}

Shape::Shape(std::vector<cv::Point> &contour)
{
    shapeContour = contour;

    float radius;
    cv::minEnclosingCircle(shapeContour, shapeCenter, radius);

    shapeArea = cv::contourArea(shapeContour);
    shapeType = Shape::classifyShape(shapeContour);

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

int Shape::classifyShape(std::vector<cv::Point> &contour)
{
    /// Approximate contour
    double perimeter = arcLength(contour, true);
    double epsilon = 0.02*perimeter;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, true);

    /// Calculate basic features
    double area = cv::contourArea(approx);
    bool isClosed = cv::isContourConvex(approx);
    int sides = approx.size();

    /// Calculate moments of contour
    cv::Moments curMnts = moments( contour, false );

    cv::Rect mbr = cv::boundingRect(contour);
    double mbrArea = mbr.area();
    double roundness = 4*PI*area/(perimeter*perimeter);
    double rectangularity = area/mbrArea;
    double eccentricity = (pow((curMnts.mu20 - curMnts.mu02),2)-4*curMnts.mu11*curMnts.mu11)/pow((curMnts.mu20 + curMnts.mu02), 2);
    double affineMomentInvariant = (curMnts.mu20*curMnts.mu02 - curMnts.mu11*curMnts.mu11)/pow(curMnts.m00,4);
    double triangularity;
    if (affineMomentInvariant <= 1.0/108) {
        triangularity = 108*affineMomentInvariant;
    } else {
        triangularity = 1/(108*affineMomentInvariant);
    }

    /// Classify shape
    int shapeType = SHAPE_NONE;
    if (area > MINIMAL_AREA && isClosed && eccentricity < 0.03) {
        if (triangularity > 0.85 && (sides >= 3 && sides <= 6)) {
            shapeType = SHAPE_TRIANGLE;

        } else if (/*(rectangularity > 0.8) &&*/ (sides == 4 || sides == 5) && (triangularity < 0.85)) {
            shapeType = SHAPE_SQUARE;

        } else if (sides == 6|| sides == 7) {
            shapeType = SHAPE_HEXAGON;

        } else if (roundness > 0.8 && sides == 8 )
            shapeType = SHAPE_CIRCLE;
    }

    return shapeType;
}

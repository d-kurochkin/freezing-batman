#include "shape.h"

//double angle_point (point a, point b, point c)
//{
//   double x1 = a.x - b.x, x2 = c.x - b.x;
//   double y1 = a.y - b.y, y2 = c.y - b.y;
//   double d1 = sqrt (x1 * x1 + y1 * y1);
//   double d2 = sqrt (x2 * x2 + y2 * y2);
//   return acos ((x1 * x2 + y1 * y2) / (d1 * d2));
//}

Shape::Shape(std::vector<cv::Point> &contour)
{
    shapeContour = contour;

    float radius;
    cv::minEnclosingCircle(shapeContour, shapeCenter, radius);

    shapeArea = cv::contourArea(shapeContour);
    shapeType = Shape::classifyShape(shapeContour);
    shapeRadius = radius;

    shapeChildrenCount += 1;
}

bool Shape::centerIsInside(std::vector<cv::Point> &contour, double &eucliadianDistance)
{

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contour, center, radius);

    double distance;
    distance = cv::pointPolygonTest(shapeContour, center, false);

    eucliadianDistance = 0;
    cv::Point diff = shapeCenter - center;
    eucliadianDistance = cv::sqrt(diff.x*diff.x + diff.y*diff.y);

    return (distance >= 0);
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

void Shape::calculateFeatures(std::vector<cv::Point> &contour, QHash<QString, QVariant> &features)
{

    /// Approximate contour
    double perimeter = arcLength(contour, true);
    double epsilon = 0.02*perimeter;
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, true);

    bool isClosed = cv::isContourConvex(approx);

    /// Calculate basic features
    double area = cv::contourArea(approx);
    double sides = (double)approx.size();

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

    features.clear();
    features.insert("perimeter", perimeter);
    features.insert("area", area);
    features.insert("sides", sides);
    features.insert("roundness", roundness);
    features.insert("rectangularity", rectangularity);
    features.insert("eccentricity", eccentricity);
    features.insert("triangularity", triangularity);
    features.insert("isClosed", isClosed);
}



int Shape::classifyShape(std::vector<cv::Point> &contour)
{
    QHash<QString, QVariant> features;
    calculateFeatures(contour, features);

    /// Classify shape
    int shapeType = SHAPE_NONE;
    if (features["area"] > MINIMAL_AREA && features["isClosed"].toBool() && features["eccentricity"] < 0.03) {
        if (features["triangularity"] > 0.85 && (features["sides"] >= 3 && features["sides"] <= 6)) {
            shapeType = SHAPE_TRIANGLE;

        } else if (/*(rectangularity > 0.8) &&*/ (features["sides"] == 4 || features["sides"] == 5) && (features["triangularity"] < 0.85)) {
            shapeType = SHAPE_SQUARE;

        } else if (features["sides"] == 6|| features["sides"] == 7) {
            shapeType = SHAPE_HEXAGON;

        } else if (features["roundness"] > 0.8 && features["sides"] == 8 )
            shapeType = SHAPE_CIRCLE;
    }

    return shapeType;
}

int Shape::detectCentralShape(cv::Mat &image, cv::Point2f center, double radius, int threshold)
{
    extern cv::Mat src;
    circle(src, center, radius,  cv::Scalar(0, 0, 0), 1);

    int count = 0;

    for (int i=0; i<360; ++i) {
        double angle = PI / 180.0 * (double)i;
        int x = (int)(center.x + radius * qCos(angle));
        int y = (int)(center.y + radius * qSin(angle));

        if (x >= 0 && y >= 0 && x < image.cols && y < image.rows) {
            int pixel = image.at<uchar>(y, x);

            if (pixel < threshold) {
                circle(src, cv::Point(x,y), 1,  cv::Scalar(0, 65, 255), 2);
                count += 1;
            }
        }
    }
    return count;
}

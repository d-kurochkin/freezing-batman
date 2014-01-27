#include "shape.h"

Shape::Shape(std::vector<cv::Point> &contour, int type)
{
    shapeContour = contour;

    float radius;
    cv::minEnclosingCircle(shapeContour, shapeCenter, radius);

    shapeArea = cv::contourArea(shapeContour);
    shapeType = type;
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


    double min_wtf = perimeter / 15;
    for (int item = 0; item < approx.size(); ++item) {

        int next;
        if(item < approx.size()-1) {
            next = item + 1;
        } else {
            next = 0;
        }

        cv::Point diff = approx[item] - approx[next];
        double eucliadianDistance = cv::sqrt(diff.x*diff.x + diff.y*diff.y);

        if(eucliadianDistance < min_wtf) {
            approx.erase(approx.begin()+item);
        }

    }


    bool isClosed = cv::isContourConvex(approx);

    /// Calculate basic features
    double area = cv::contourArea(approx);
    double sides = (double)approx.size();

    /// Calculate moments of contour
    cv::Moments curMnts = moments(approx, false );

    cv::RotatedRect mbr = cv::minAreaRect(approx);
    double mbrArea = mbr.size.area();
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



int Shape::classifyShape(std::vector<cv::Point> &contour, double threshold)
{
    QHash<QString, QVariant> features;
    calculateFeatures(contour, features);

    const double featuresValues[4] = {
        features["roundness"].toDouble(),
        features["rectangularity"].toDouble(),
        features["triangularity"].toDouble(),
        features["sides"].toDouble()/8.0,
    };

    double resultDistance[4] = {0};

    /// Classify shape
    int shapeType = SHAPE_NONE;
    if (features["area"] > MINIMAL_AREA && features["isClosed"].toBool() && features["eccentricity"] < 0.03) {
        for (int shape_index=0; shape_index<4; ++shape_index) {
            for (int feature_index=0; feature_index<4; ++feature_index) {
                resultDistance[shape_index] += qAbs(featuresValues[feature_index] - prototypesFeatures[shape_index][feature_index]);
            }
        }
        double minValue = 10;
        double minIndex = 0;
        for(int i=0; i<4; ++i) {
            if (resultDistance[i] < minValue) {
                minValue = resultDistance[i];
                minIndex = i;
            }
        }
        if (minValue <= threshold) {
            //qDebug() << minIndex+1 << " -> " << minValue;
            shapeType = SHAPE_NONE + minIndex + 1;
        }

    } else {
        shapeType = SHAPE_NONE;
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

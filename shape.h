#ifndef SHAPE_H
#define SHAPE_H

#include <QDebug>
#include <QHash>
#include <QVariant>
#include <QString>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum {
    SHAPE_NONE      = 0,
    SHAPE_TRIANGLE  = 1,
    SHAPE_SQUARE    = 2,
    SHAPE_HEXAGON   = 3,
    SHAPE_CIRCLE    = 4,
    SHAPE_CENTER    = 5,
    SHAPE_MAIN      = 6
};

/// Define shapes colors

const cv::Scalar SHAPE_COLORS[7] = {
    cv::Scalar(0, 0, 0),
    cv::Scalar(28, 232, 0),
    cv::Scalar(28, 247, 255),
    cv::Scalar(212, 96, 0),
    cv::Scalar(28, 0, 255),
    cv::Scalar(255, 255, 255),
    cv::Scalar(255, 255, 255)
};

const int MINIMAL_AREA = 200;
const double PI = 3.141592653589793238462;

class Shape
{
    public:
        std::vector<cv::Point> shapeContour;
        cv::Point2f shapeCenter;
        float shapeArea = 0;
        int shapeChildrenCount = 0;
        int shapeType = SHAPE_NONE;

    public:
        Shape(){}
        Shape(std::vector<cv::Point> &contour);
        bool centerIsInside(std::vector<cv::Point> &contour, double &eucliadianDistance);
        void mergeContours(std::vector<cv::Point> &contour);

    public:
        static void calculateFeatures(std::vector<cv::Point> &contour, QHash<QString, QVariant> &features);
        static int classifyShape(std::vector<cv::Point> &contour);

};

#endif // SHAPE_H

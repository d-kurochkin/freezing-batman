#ifndef SHAPE_H
#define SHAPE_H

#include <QDebug>
#include <QHash>
#include <QVariant>
#include <QString>
#include <QtMath>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

enum {
    SHAPE_NONE      = 0,
    SHAPE_TRIANGLE  = 1,
    SHAPE_SQUARE    = 2,
    SHAPE_HEXAGON   = 3,
    SHAPE_CIRCLE    = 4,
    SHAPE_CENTER    = 5,
    SHAPE_PLATFORM  = 6
};

/// Define shapes colors

const cv::Scalar SHAPE_COLORS[7] = {
    cv::Scalar(0, 0, 0),
    cv::Scalar(28, 232, 0),
    cv::Scalar(28, 247, 255),
    cv::Scalar(212, 96, 0),
    cv::Scalar(28, 0, 255),
    cv::Scalar(255, 222, 63),
    cv::Scalar(255, 255, 255)
};

const int MINIMAL_AREA = 200;
const double PI = 3.141592653589793238462;


/// Порядок следования признаков: Roundness, Rectangularity, Triangularity, Количество углов
const double prototypesFeatures[4][4] = {
    {0.56,  0.51,    1,     0.375},
    {0.79,  1,      0.75,   0.5},
    {0.83,  0.75,   0.69,   0.75},
    {0.9,   0.77,   0.68,   1},
};


const double prototypesFactors[4][4] = {
    {1,     1,      1.5,    0.8},
    {1,     1.5,    1,      1},
    {1,     1,      1,      1},
    {1.5,   1,      1,      1},
};

class Shape
{
    public:
        std::vector<cv::Point> shapeContour;
        cv::Point2f shapeCenter;
        float shapeArea = 0;
        float shapeRadius = 0;
        int shapeChildrenCount = 0;
        int shapeType = SHAPE_NONE;

    public:
        Shape(){}
        Shape(std::vector<cv::Point> &contour, int type);
        bool centerIsInside(std::vector<cv::Point> &contour, double &eucliadianDistance);
        void mergeContours(std::vector<cv::Point> &contour);

    public:
        static void calculateFeatures(std::vector<cv::Point> &contour, QHash<QString, QVariant> &features);
        static int classifyShape(std::vector<cv::Point> &contour, double threshold);
        static int detectCentralShape(cv::Mat &image, cv::Point2f center, double radius, int threshold);

};

#endif // SHAPE_H

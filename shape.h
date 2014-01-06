#ifndef SHAPE_H
#define SHAPE_H

#include <opencv2/core/core.hpp>

enum {
    SHAPE_NONE      = 0,
    SHAPE_TRIANGLE  = 1,
    SHAPE_SQUARE    = 2,
    SHAPE_HEXAGON   = 3,
    SHAPE_CIRCLE    = 4,
    SHAPE_CENTER    = 5,
    SHAPE_MAIN      = 6
};

class Shape
{
    public:
        std::vector<cv::Point> contour;
        cv::Point center;
        float area = 0;
        int childrenCount = 0;
    public:
        Shape();
        bool centerIsInside(const std::vector<cv::Point>);
        void mergeContours(const std::vector<cv::Point>);

};

#endif // SHAPE_H

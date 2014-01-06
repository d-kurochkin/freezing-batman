#-------------------------------------------------
#
# Project created by QtCreator 2014-01-06T01:23:26
#
#-------------------------------------------------
CONFIG += c++11

QT       += core

QT       -= gui

TARGET = freezing-batman
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    shape.cpp

HEADERS += \
    shape.h


# add opencv
INCLUDEPATH += C:\Development\opencv\include


LIBS +=-LC:\Development\opencv\lib\
-lopencv_calib3d248\
-lopencv_contrib248\
-lopencv_core248\
-lopencv_features2d248\
-lopencv_flann248\
-lopencv_gpu248\
-lopencv_highgui248\
-lopencv_imgproc248\
-lopencv_legacy248\
-lopencv_ml248\
-lopencv_nonfree248\
-lopencv_objdetect248\
-lopencv_ocl248\
-lopencv_photo248\
-lopencv_stitching248\
-lopencv_superres248\
-lopencv_video248\
-lopencv_videostab248\

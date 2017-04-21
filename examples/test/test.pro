TEMPLATE = app
QT = gui
CONFIG += debug console

HEADERS =
SOURCES = main.cpp

INCLUDEPATH += ../../src 
LIBS += -lVOCHOG -L../../src -L/usr/local/cuda/lib64 -lQtCore -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

DESTDIR = ../../bin

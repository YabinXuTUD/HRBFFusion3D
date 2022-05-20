#ifndef WINDOW_H
#define WINDOW_H

#include <QWidget>
#include "lineslam.h"


class QSlider;

class GLWidget;

class Window : public QWidget
{
	Q_OBJECT

public:
	Window();
	void setScene(Map3d* m);
	
protected:
	void keyPressEvent(QKeyEvent *event);

private:
	QSlider *createSlider();
	QSlider *createScaleSlider();

	GLWidget *glWidget;
	QSlider *xSlider;
	QSlider *ySlider;
	QSlider *zSlider;
	QSlider *sSlider; // scale
};

#endif

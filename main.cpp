#include <iostream>
#include "HoughCirclesDetection.h"
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    auto lImg = cv::imread("eye2.jpg");
    cv::resize(lImg, lImg, cv::Size(), 2.0, 2.0);
	cv::cvtColor(lImg, lImg, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> lChannels;
	cv::split(lImg, lChannels);
	auto lIntensity = lChannels[2];
	cv::medianBlur(lIntensity, lIntensity, 5);

	double lMinVal;
	cv::minMaxLoc(lIntensity, &lMinVal);
	lMinVal += 30.0;
	
	cv::Mat lMask;
	auto lKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, { 5, 5 });
	cv::threshold(lIntensity, lMask, lMinVal, 255.0, cv::THRESH_BINARY_INV);
	cv::erode(lMask, lMask, lKernel);
	cv::dilate(lMask, lMask, lKernel);

    cv::Mat lEdges;
    cv::Canny(lMask, lEdges, 50, 100);
	auto lSubEdges = lEdges; 

	cv::imshow("Mask", lMask);
	cv::imshow("Intensity", lIntensity);
	cv::imshow("SubEdge", lSubEdges);

	std::vector<float> lRadii(20);
	auto lStart = 4.0f;
	for (auto i = 0; i < lRadii.size(); i++)
	{
		lRadii[i] = lStart + i * 1.0f;
	}
    CircleAccumulator lAcc(lSubEdges.cols, lSubEdges.rows, lRadii, 2.0, 10.0);
    lAcc.accumulate(lSubEdges);

	cv::cvtColor(lImg, lImg, cv::COLOR_HSV2BGR);
    auto lCircles = lAcc.findArgmax(5, 4, 1, 0.0, 30, CircleAccumulator::Normalization::LevelWise);
	auto i = 0;
	cv::Mat lImgDraw;
	lImg.copyTo(lImgDraw);
    for (auto lCircle : lCircles)
    {
		if (i == 0)
		{
			cv::circle(lImgDraw, { static_cast<int>(lCircle[0]), static_cast<int>(lCircle[1]) }, lCircle[2],
				cv::Scalar(0, 255, 0));
			i++;
		}
		else
		{
			cv::circle(lImgDraw, { static_cast<int>(lCircle[0]), static_cast<int>(lCircle[1]) }, lCircle[2],
				cv::Scalar(0, 0, 255));
		}
    }
    cv::imshow("Home-made", lImgDraw);
    cv::waitKey();

    return 0;
}

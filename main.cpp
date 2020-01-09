#include <iostream>
#include "HoughCirclesDetection.h"
#include <numeric>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main() {
    auto lImg = cv::imread("test.png");

    cv::Mat lEdges;
    cv::Canny(lImg, lEdges, 50, 200);
    auto lSubEdges = lEdges.rowRange(150, 500).colRange(450, 750);
	// auto lSubEdges = lEdges;

	cv::imshow("SubEdge", lSubEdges);
	cv::waitKey();

	std::vector<float> lRadii(100);
	auto lStart = 20.0f;
	for (auto i = 0; i < lRadii.size(); i++)
	{
		lRadii[i] = lStart + i * 2.5f;
	}
    CircleAccumulator lAcc(lSubEdges.cols, lSubEdges.rows, lRadii, 1.0, 10.0);
    lAcc.accumulate(lSubEdges);

	// Debug
	/*
	for (auto i = 0; i < lAcc.getNLevels(); i++)
	{
		auto lStrRadius = std::to_string(lAcc.getRadius(i));
		lStrRadius = lStrRadius.substr(0, lStrRadius.find('.') + 3);
		cv::imshow("AccLvl" + lStrRadius, lAcc.getLevel(i) * 255);
	}
	cv::waitKey();
	*/

    auto lCircles = lAcc.findArgmax(2, 4, 0.0, 0.1, 10, -1);
    for (auto lCircle : lCircles)
    {
        // cv::circle(lImg, {static_cast<int>(lCircle[0]), static_cast<int>(lCircle[1])}, lCircle[2],
        //         cv::Scalar(0, 0, 255));

		cv::circle(lImg, { static_cast<int>(lCircle[0]) + 450, static_cast<int>(lCircle[1]) + 150 }, lCircle[2],
			cv::Scalar(0, 0, 255));
    }
    cv::imshow("Image", lImg);
    cv::waitKey();

    return 0;
}

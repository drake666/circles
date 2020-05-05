#ifndef CIRCLES_HOUGHCIRCLESDETECTION_H
#define CIRCLES_HOUGHCIRCLESDETECTION_H

#include <vector>
#include <array>
#include <opencv2/core.hpp>


using Circle = std::array<float, 3>;  /// x, y, r
using Peak = std::array<float, 3>;    /// x, y, v
using PeakR = std::array<float, 4>;    /// x, y, v, r

class CircleAccumulator
{
public:
    enum class Normalization
    {
        NoNormalization = -1,
        LevelWise,
        CircleCircumference
    };

    CircleAccumulator(int pWidth, int pHeight, std::vector<float> pRadii, float pXYStep=1.0, float pThetaStep=1.0);

    /// Must be the same size as (width, height) / minDistance
    void accumulate(cv::Mat pEdges);

    /// The default parameter return only 1 detection, the best
    std::vector<Circle> findArgmax(int pMaxNbPos=1, int pMaxPeak=1, float pMinDistance=1.0, float pThreshold=1.0,
                                   int pAbsThreshold=0, Normalization pNormalize = Normalization::LevelWise);

    const cv::Mat getLevel(int pLevel) const;

    const int getNLevels() const;
    const float getRadius(int pLevel) const;

private:
    std::vector<cv::Mat> mLevels;
    std::vector<float> mRadii;
    float mXYStep;
    float mThetaStep;
};


#endif //CIRCLES_HOUGHCIRCLESDETECTION_H

#define _USE_MATH_DEFINES

#include "HoughCirclesDetection.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <set>
#include <iostream>
#include <queue>


CircleAccumulator::CircleAccumulator(int pWidth, int pHeight, std::vector<float> pRadii, float pXYStep,
                                     float pThetaStep)
: mRadii(pRadii), mXYStep(pXYStep), mThetaStep(pThetaStep)
{
    assert(!pRadii.empty() && pXYStep > 0 && pThetaStep > 0);
    mLevels.reserve(pRadii.size());
    for (auto r : pRadii)
    {
        mLevels.push_back(cv::Mat::zeros(int(pHeight / pXYStep), int(pWidth / pXYStep), CV_16U));
    }
}

void CircleAccumulator::accumulate(cv::Mat pEdges)
{
    assert(!pEdges.empty());
    auto lWidth = mLevels[0].cols;
    auto lHeight = mLevels[0].rows;

    for (auto i = 0; i < pEdges.cols; i++)
    {
        for (auto j = 0; j < pEdges.rows; j++)
        {
            if (pEdges.at<std::uint8_t>(j, i) == 0)
            {
                continue;
            }
            for (auto k = 0; k < mRadii.size(); k++)
            {
                auto r = mRadii[k];
                for (auto theta = 0.0f; theta < 360.0f; theta += mThetaStep)
                {
                    auto a = int((i - r * std::cos(theta * M_PI / 180.0)) / mXYStep); //polar coordinate for center
                    auto b = int((j - r * std::sin(theta * M_PI / 180.0)) / mXYStep);  //polar coordinate for center
                    if (a >= 0 && b >= 0 && a < lWidth && b < lHeight)
                    {
                        mLevels[k].at<std::uint16_t>(b, a) += 1; //voting
                    }
                }
            }
        }
    }
}

struct peakLess : public std::binary_function<Peak, Peak, bool>
{
    bool
    operator()(const Peak& __x, const Peak& __y) const
    { return __x[2] < __y[2]; }
};

struct peakRLess : public std::binary_function<PeakR, PeakR, bool>
{
    bool
        operator()(const PeakR& __x, const PeakR& __y) const
    {
        return __x[2] < __y[2];
    }
};


std::vector<Circle> CircleAccumulator::findArgmax(int pMaxNbPos, int pMaxPeak, float pMinDistance,
                                                  float pThreshold, int pAbsThreshold, Normalization pNormalize)
{
    std::vector<Circle> lDetections = {};
    std::vector<int> lLevelTotal(mLevels.size(), 0);
    std::vector<std::vector<Peak>> lLevelPeaks(mLevels.size());
    std::priority_queue<PeakR, std::vector<PeakR>, peakRLess> lGreedyPeaks;

    if (pNormalize == Normalization::LevelWise)
    {
        // Level-wise normalization
        for (auto k = 0; k < mRadii.size(); k++)
        {
            lLevelTotal[k] = cv::sum(mLevels[k])[0];
        }
    }
    else if (pNormalize == Normalization::CircleCircumference)
    {
        // Circle circumference
        for (auto k = 0; k < mRadii.size(); k++)
        {
            lLevelTotal[k] = std::ceil(2.0f * static_cast<float>(M_PI) * (mRadii[k] - 1) / mThetaStep);
        }
    }
    else
    {
        // No normalization
        for (auto k = 0; k < mRadii.size(); k++)
        {
            lLevelTotal[k] = 1;
        }
    }

    // Multi-level argmax
    for (auto i = 0; i < mLevels[0].cols; i++)
    {
        for (auto j = 0; j < mLevels[0].rows; j++)
        {
            for (auto k = 0; k < mRadii.size(); k++)
            {
                const auto& lVal = mLevels[k].at<std::uint16_t>(j, i);
                auto lNormalizeValue = static_cast<float>(lVal) / static_cast<float>(lLevelTotal[k]);
                if (lVal > pAbsThreshold && lNormalizeValue > pThreshold)
                {
                    // x, y, v
                    lLevelPeaks[k].push_back({static_cast<float>(i) * mXYStep, static_cast<float>(j) * mXYStep, lNormalizeValue});
                    lGreedyPeaks.push({static_cast<float>(i) * mXYStep, static_cast<float>(j) * mXYStep, lNormalizeValue, mRadii[k]});
                }
            }
        }
    }

    // What I should do:
    // 1. Merge peak XY-wise (hierachical clustering style)
    // 2. Merge peak radii (same way)
    // 3. Keep pMaxNbPos; keep only location that the best level is superior to the threshold
    // 4. In those location, keep pMaxPeak per Pos; keep only peaks which are supérior to the threshold

    // Instead: let's do it REALLY greedy
    auto lSqMinDistance = pMinDistance * pMinDistance;
    while (!lGreedyPeaks.empty() && lDetections.size() != pMaxNbPos)
    {
        auto lPeak = lGreedyPeaks.top();
        lGreedyPeaks.pop();

        auto lToAdd = true;
        auto lIdx = -1;
        auto lMinDist = 1e10f;
        for (auto i = 0; i < lDetections.size(); i++)
        {
            const auto& lDetection = lDetections[i];
            auto lDist = std::pow(lDetection[0] - lPeak[0], 2) + std::pow(lDetection[1] - lPeak[1], 2);
            if (std::pow(lDetection[0] - lPeak[0], 2) + std::pow(lDetection[1] - lPeak[1], 2) < lSqMinDistance && lMinDist > lDist)
            {
                lToAdd = false;
                break;
            }
        }
        if (lToAdd)
        {
            lDetections.push_back({ lPeak[0], lPeak[1], lPeak[3] });
        }
    }

    return lDetections;
}

const cv::Mat CircleAccumulator::getLevel(int pLevel) const
{
    if (pLevel < 0 || pLevel >= mLevels.size())
    {
        std::cout << "WARNING: ask to show accumulator level " << pLevel << " which does not exist (" << mLevels.size() << ")" << std::endl;
        return cv::Mat();
    }
    return mLevels[pLevel];
}


const float CircleAccumulator::getRadius(int pLevel) const
{
    return mRadii[pLevel];
}


const int CircleAccumulator::getNLevels() const
{
    return mLevels.size();
}

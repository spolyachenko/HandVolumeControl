/**@file PalmDetector.cpp
 * @brief Definition of functions for converting landmarks to RoI crops
 */

#pragma once

#include <iostream>
#include <vector>

namespace cv
{
    class Mat;
    class RotatedRect;
}

namespace Ort
{
    struct Value;
}

namespace HandLandmarker
{   
    /**@brief Takes in a batch of crop parameters as list of cv::RotatedRect and converts them to source image crops
     * 
     * @param crop_rects Vector of rectangles to be cropped
     * @param src_img Source image
     * @param landmarker_input_width Target Width of crop
     * @param landmarker_input_height Target Height of crop
     * @param crops Where to put Images of crops
     */
    void detection_to_roi_crop(
        const std::vector<cv::RotatedRect> &crop_rects, 
        const cv::Mat &src_img,
        const int landmarker_input_width, 
        const int landmarker_input_height,
        std::vector<cv::Mat> &crops
    );

    /**@brief Decodes Landmarker outputs into both source image coordinates and normalized [0, 1] range coordinates
     * @param outputs Landmarker outputs, 3 vectors: Conf score, Handedness and 21 x,y,z in [0;1] crop rectangle coords
     * @param roi_rect RotatedRect of crop, used to decode landmark coordinates
     * @return pair of vectors size 2 + 21*3 = 65, {conf, handedness, x1, y1, z1, ... , x21, y21, z21}. First vector's x, y in src image coords, Second vector's coords are normalized to [0;1]
     */
    std::pair<std::vector<float>, std::vector<float>> parse_landmarker_outputs(
        const std::vector<Ort::Value> &outputs, 
        const cv::RotatedRect &roi_rect
    );

    /**@brief Converts Lanmdarks into next frame RoI RotatedRect and image crop
     * @param landmarks Landmarks vector size 65, {score, handedness, x1, y1, z1, ... , x21, y21, z21}. x, y in src coords
     * @param src_img Source image
     * @param landmarker_input_width Target Width of crop
     * @param landmarker_input_height Target Height of crop
     * @return pair of crop image and rotated rectangle for next frame's tracking RoI
     */
    std::pair<cv::Mat, cv::RotatedRect> landmarks_to_roi_crop(
        const std::vector<float> &landmarks, 
        const cv::Mat &src_img, 
        const int landmarker_input_width, 
        const int landmarker_input_height
    );
}
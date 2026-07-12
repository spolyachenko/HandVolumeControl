/**@file PalmDetector.hpp
 * @brief Declaration of Functions for processing Palm Detector outputs
 * 
 * Provides two functions:
 * 
 * - Preload anchors from .npy file
 * 
 * - Convert ONNX ouputs to bounding boxes of detected palms
 */

#pragma once

#include <string>
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

namespace PalmDetector
{   
    /** @brief Preloads anchor boxes from a numpy .npy file and puts them into a flattened vector
     * 
     * For palm detections.npy file assumed to consist of ndarray [num_anchors, 4] w dtype=double. Each anchor is given by (x_center, y_center, width, height)
     * 
     * @param path Path to numpy .npy file
     * @param anchors Flat vector of parsed anchor's parameters. 
     * @return Void
     */
    void preload_anchors(const std::string &path, std::vector<double> &anchors);

    /** @brief Processes palm detection outputs
     * 
     * Converts detector's outputs to bboxes and obtains rotated crops. Rotation makes it so the hand palm is vertical in the cropped image.
     * 
     *For each pre-defined anchor box, palm detector ouputs:

     * - Box's detection confidence score

     * - SSD style anchor to bbox offsets: dx, dy, dw, dh

     * - 7 palm keypoints
     * 
     * Conversion from anchor box (a_x, a_y, a_w, a_h) to bounding box (xc, yc, w, h) for input image of size (W, H) is given by:
     * 
     * xc = dx * a_w / W + a_x
     * 
     * yc = dy * a_h / H + a_y
     * 
     * w  = dw * a_w / W
     * 
     * h  = dh * a_h / H
     * 
     * @param outputs Detector's outputs
     * @param anchors Preloaded anchors
     * @param trg_width Crop width
     * @param trg_height Crop height
     * @param conf_threshold Confidence score threshold to filter detections 
     * @param iou_threshold Intersection over Union threshold to consider 2 detections as overlapping
     * @return Flat vector of RotatedRect parameters (xc, yc, w, h, angle). 
     * Center point, width and height in relative img coordinates [0, 1], angle is given in radians 
     */
    std::vector<cv::RotatedRect> process_detector_outputs(
        const std::vector<Ort::Value> &outputs, 
        const std::vector<double> &anchors,
        const int detector_input_width, 
        const int detector_input_height,
        const int src_img_width, 
        const int src_img_height,
        cv::Mat &src_img,
        const float conf_threshold=0.5, 
        const float iou_threshold=0.3
    );
}
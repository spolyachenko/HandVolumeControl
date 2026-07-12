/**@file PalmDetector.cpp
 * @brief Implementation of functions for processing Palm Detector outputs
 */

#include "PalmDetector.hpp"
#include "npy.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>
#include <set>
#include <numbers>

namespace {
    /** @brief Struct used to contain Palm detector outputs
     * 
     * Each detection consists of:
     * 
     * - Bounding box, given by center point (xc, yc), width w and height h
     * 
     * - 7 palm landmark points, given by vector {x_1, y_1, x_2, y_2, ..., x_7, y_7}
     * 
     * - Confidence score for this detection
     */
    struct PalmDetections 
    {
        float xc{0.0f}, yc{0.0f}, w{0.0f}, h{0.0f}, score{0.0f};
        std::vector<float> landmarks;

        
        /** @brief Default constructor for PalmDetections.
         * 
         * Initializes all 18 detection outputs to 0.
         */
        PalmDetections(): xc{0}, yc{0}, w{0}, h{0}, score{0} {
            for (size_t i = 0; i < 2*7; i++)
            {
                landmarks.push_back(0);
            }
        }

        PalmDetections(float _xc, float _yc, float _w, float _h, float _score): xc{_xc}, yc{_yc}, w{_w}, h{_h}, score{_score} 
        {}

        /** @brief Compute IoU for 2 detections using their bbox parameters.
         *  
         * @param other Other detection, which bbox to use for IoU.
         * @return Intersection over Union of given detections bounding boxes
         */
        float IoU(const PalmDetections &other) const
        {
            float self_x_min = this->xc - 0.5f * this->w;
            float self_y_min = this->yc - 0.5f * this->h;
            float self_x_max = this->xc + 0.5f * this->w;
            float self_y_max = this->yc + 0.5f * this->h;

            float other_x_min = other.xc - 0.5f * other.w;
            float other_y_min = other.yc - 0.5f * other.h;
            float other_x_max = other.xc + 0.5f * other.w;
            float other_y_max = other.yc + 0.5f * other.h;

            float inter_x_min = std::max(self_x_min, other_x_min);
            float inter_y_min = std::max(self_y_min, other_y_min);
            float inter_x_max = std::min(self_x_max, other_x_max);
            float inter_y_max = std::min(self_y_max, other_y_max);

            float inter_area = std::max(0.0f, inter_x_max - inter_x_min) * std::max(0.0f, inter_y_max - inter_y_min);
            float union_area = this->h * this->w + other.h * other.w - inter_area;
            
            return inter_area / union_area;
        }
    };

    bool operator<(const PalmDetections &a, const PalmDetections &b)
    {
        return a.score > b.score;
    }

    /** @brief Per-parameter sum of two detections
     * 
     * Added detection gets weighted by its confidence score.
     * 
     * Used in Weighted Non-max suppression to average overlapping predictions, weighted by detection's confidence score.
     * 
     * @param src Source detection object
     * @param other Weighted detection to be added
     */
    void add_detections(PalmDetections &src, const PalmDetections &other)
    {
        src.xc += other.score * other.xc;
        src.yc += other.score * other.yc;
        src.w += other.score * other.w;
        src.h += other.score * other.h;
        src.score += other.score;

        for (size_t i = 0; i < src.landmarks.size(); i++)
        {
            src.landmarks[i] += other.score * other.landmarks[i];
        }
    }

    /** @brief Normalize each detection output by number of overlaps
     * 
     * Used with .add_detection to calculate weighted average of detection overlaps
     * 
     * @param src Source detection to be normalized
     * @param overlap Number of used overlaps
     */
    void normalize_detection(PalmDetections &src, int overlap_count)
    {
        src.xc /= src.score;
        src.yc /= src.score;
        src.h /= src.score;
        src.w /= src.score;
        
        for (size_t i = 0; i < src.landmarks.size(); i++)
        {
            src.landmarks[i] /= src.score;
        }
        
        src.score /= overlap_count;
    }

    /** @brief Weighted Non-max suppression algorithm
     * 
     * Standard NMS algorithm, except overlapping detections get averaged weighted by their confidence score. 
     * 
     * Resulting detection's confidence score is the mean of overlapping detection's scores.
     * 
     * @param detections Multiset of input detections
     * @param weighted_detections Suppressed and weighted detections
     * @param conf_threshold Confidence score threshold. Detections should come pre-filtered by this, also filters resulting weighted detections
     * @param iou_threshold Intersection over Union threshold to consider 2 detections as overlapping.
     */
    void weighted_non_max_suppression(std::multiset<PalmDetections> &detections, 
        std::vector<PalmDetections> &weighted_detections, float conf_threshold, float iou_threshold)
    {
        while(!detections.empty()) 
        {
            PalmDetections base_detection = *detections.begin();
            PalmDetections weighted_detection;

            int overlap_count{0};

            for (auto it = detections.begin(); it != detections.end(); ) {
                const PalmDetections &other_detection = *it;

                float iou = base_detection.IoU(other_detection);

                float dx = base_detection.xc - other_detection.xc;
                float dy = base_detection.yc - other_detection.yc;

                if (iou > iou_threshold)
                {
                    add_detections(weighted_detection, other_detection);
                    overlap_count++;
                    it = detections.erase(it);
                }
                else
                {
                    it++;
                }
            }

            normalize_detection(weighted_detection, overlap_count);
            weighted_detections.push_back(weighted_detection);
        }
    }
}

namespace PalmDetector
{
    void preload_anchors(const std::string &path, std::vector<double> &anchors)
    {
        npy::npy_data d = npy::read_npy<double>(path);
        std::vector<unsigned long> shape = d.shape;
        anchors = d.data;
    }

    std::vector<cv::RotatedRect> process_detector_outputs(const std::vector<Ort::Value> &outputs, const std::vector<double> &anchors,
        const int detector_input_width, const int detector_input_height,
        const int src_img_width, const int src_img_height, cv::Mat &src_img,
        const float conf_threshold, const float iou_threshold
    )
    {   
        assert(outputs.size() == 2);

        std::vector<int64_t> box_coords_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const float *box_coords_ptr = outputs[0].GetTensorData<float>();

        std::vector<int64_t> box_scores_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const float *box_scores_ptr = outputs[1].GetTensorData<float>();

        std::multiset<PalmDetections> detections;
        std::vector<PalmDetections> palm_detections;

        int64_t pred_offset = box_coords_shape[2];

        for (size_t i = 0; i < box_scores_shape[1]; i++)
        {   
            float score = 1.0 / (1.0 + std::exp(-*(box_scores_ptr + i)));
            
            if (score < conf_threshold) {continue;}

            float anchor_x = anchors[4 * i + 0];
            float anchor_y = anchors[4 * i + 1];
            float anchor_w = anchors[4 * i + 2];
            float anchor_h = anchors[4 * i + 3];

            float dx = *(box_coords_ptr + pred_offset*i + 0);
            float dy = *(box_coords_ptr + pred_offset*i + 1);
            float dw = *(box_coords_ptr + pred_offset*i + 2);
            float dh = *(box_coords_ptr + pred_offset*i + 3);
            
            float xc = dx / detector_input_width * anchor_w + anchor_x;
            float yc = dy / detector_input_height * anchor_h + anchor_y;
            float w = dw / detector_input_width * anchor_w;
            float h = dh / detector_input_height * anchor_h;
            
            PalmDetections detection(xc, yc, w, h, score);
            
            for (size_t j = 4; j < pred_offset; j+=2)
            {   
                float x{*(box_coords_ptr + pred_offset*i + j)};
                float y{*(box_coords_ptr + pred_offset*i + j + 1)};
                
                x = (x / detector_input_width * anchor_w + anchor_x);
                y = (y / detector_input_height * anchor_h + anchor_y);
                detection.landmarks.push_back(x);
                detection.landmarks.push_back(y);
            }

            detections.insert(detection);
        }
        
        weighted_non_max_suppression(detections, palm_detections, conf_threshold, iou_threshold);

        std::sort(palm_detections.begin(), palm_detections.end(), [](PalmDetections &a, PalmDetections &b){
            return std::max(a.h, a.w) > std::max(b.h, b.w);
        });
        
        const float dy{-0.3f};
        const int Wrist{0};
        const int IndMCP{1};
        const int MidMCP{2};
        const int RingMCP{3};
        const int PinkyMCP{4};

        std::vector<cv::RotatedRect> crop_rects;

        for (const PalmDetections &det: palm_detections)
        {   
            for (size_t i = 0; i < 7; i++)
            {   
                const float x = det.landmarks[2*i] * src_img_width;
                const float y = det.landmarks[2*i + 1] * src_img_height;

                cv::circle(src_img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
            
            const float x0 = det.landmarks[2 * Wrist + 0] * src_img_width;
            const float y0 = det.landmarks[2 * Wrist + 1] * src_img_height;

            const float x1 = det.landmarks[2 * MidMCP + 0] * src_img_width;
            const float y1 = det.landmarks[2 * MidMCP + 1] * src_img_height;

            const float rot_angle = std::atan2(y0 - y1, x0 - x1) - std::numbers::pi / 2.0f;

            float scale = std::max(det.w * src_img_width, det.h * src_img_height);

            float xc = det.xc * src_img_width;
            float yc = det.yc * src_img_height;

            yc += dy * scale * std::cos(rot_angle);
            xc -= dy * scale * std::sin(rot_angle);

            scale *= 2.2f;

            cv::RotatedRect roi_rect(
                cv::Point2f(xc, yc),
                cv::Size2f(scale, scale),
                rot_angle * 180.f / std::numbers::pi
            );

            crop_rects.push_back(roi_rect);
        }

        return crop_rects;
    }
} // namespace PalmDetector
#include "HandLandmarker.hpp"

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <numbers>
#include <iostream>

namespace HandLandmarker
{   
    void detection_to_roi_crop(
        const std::vector<cv::RotatedRect> &crop_rects, 
        const cv::Mat &src_img,
        const int landmarker_input_width, 
        const int landmarker_input_height,
        std::vector<cv::Mat> &crops
    )
    {   
        const cv::Size trg_sz(landmarker_input_width, landmarker_input_height);
        const cv::Size src_sz = src_img.size();

        for (size_t i = 0; i < crop_rects.size(); i++)
        {
            const cv::RotatedRect &rotated_rect = crop_rects[i];
                        
            cv::Mat transform_matrix = cv::getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);
            
            transform_matrix.at<double>(0, 2) += (0.5f * rotated_rect.size.width - rotated_rect.center.x);
            transform_matrix.at<double>(1, 2) += (0.5f * rotated_rect.size.height - rotated_rect.center.y);
            
            cv::Mat crop;
            cv::warpAffine(src_img, crop, transform_matrix, rotated_rect.size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            cv::resize(crop, crop, trg_sz, 0.0, 0.0, cv::INTER_LINEAR);

            crops.push_back(crop);
        }
    }

    std::pair<cv::Mat, cv::RotatedRect> landmarks_to_roi_crop(
        const std::vector<float> &landmarks, 
        const cv::Mat &src_img, 
        const int landmarker_input_width, 
        const int landmarker_input_height
    )
    {   
        assert (landmarks.size() == 65);
        const cv::Size img_size = src_img.size();
        
        const int Wrist{0};
        const int Thumb_CMC{1};

        const int Ind_MCP{5};
        const int Mid_MCP{9};
        const int Ring_MCP{13};
        const int Pinky_MCP{17};

        const int Ind_PIP{6};
        const int Mid_PIP{10};
        const int Ring_PIP{14};
        const int Pinky_PIP{18};


        // Calculate angle based on wrist and averaged finger mcp's
        float x0 = landmarks[2 + 3 * Wrist + 0];
        float y0 = landmarks[2 + 3 * Wrist + 1];

        float x1 = (landmarks[2 + 3 * Ind_MCP + 0] + landmarks[2 + 3 * Ring_MCP + 0]) / 2.f;
        float y1 = (landmarks[2 + 3 * Ind_MCP + 1] + landmarks[2 + 3 * Ring_MCP + 1]) / 2.f;

        x1 = (x1 + landmarks[2 + 3 * Mid_MCP + 0]) / 2.f;
        y1 = (y1 + landmarks[2 + 3 * Mid_MCP + 1]) / 2.f;

        float dx = x1-x0;
        float dy = y1-y0;

        const float rot_angle = std::atan2(y0 - y1, x0 - x1) - std::numbers::pi / 2.0f;

        // Calculate center point as average of MCP and PIP points
        float x2 = (landmarks[2 + 3 * Ind_PIP + 0] + landmarks[2 + 3 * Ring_PIP + 0]) / 2.f;
        float y2 = (landmarks[2 + 3 * Ind_PIP + 1] + landmarks[2 + 3 * Ring_PIP + 1]) / 2.f;

        x2 = (x2 + landmarks[2 + 3 * Mid_PIP + 0]) / 2.f;
        y2 = (y2 + landmarks[2 + 3 * Mid_PIP + 1]) / 2.f;

        x1 = (x1 + x2) * 0.5f;
        y1 = (y1 + y2) * 0.5f;

        // Shift center point towards wrist
        const float k = 0.2f;
        x1 -= dx * k;
        y1 -= dy * k;

        float scale = 0.f;
        const std::vector<int> rigid = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
        int i_max{0}, j_max{0};
        for (size_t i = 0; i < rigid.size(); i++)
        {
            float xi = landmarks[2 + 3 * rigid[i] + 0];
            float yi = landmarks[2 + 3 * rigid[i] + 1];

            if (i == 0)
            {
                xi = xi * 0.7f + 0.3f * landmarks[2 + 3 * 1 + 0];
                yi = yi * 0.7f + 0.3f * landmarks[2 + 3 * 1 + 1];
            }

            for (size_t j = i+1; j < rigid.size(); j++)
            {
                float xj = landmarks[2 + 3 * rigid[j] + 0];
                float yj = landmarks[2 + 3 * rigid[j] + 1];
                float dist = std::sqrt((xj-xi)*(xj-xi) + (yj-yi)*(yj-yi));
                
                if (scale < dist) {
                    scale = dist;
                    i_max = rigid[i];
                    j_max = rigid[j];
                }
            }
        }
        scale *= 2.2f;

        cv::RotatedRect roi_rect(
            cv::Point2f(x1, y1), 
            cv::Size2f(scale, scale), 
            rot_angle * 180.0f / std::numbers::pi
        );
        
        // Obtain image crop from src
        std::vector<cv::Mat> crop;
        detection_to_roi_crop({roi_rect}, src_img, landmarker_input_width, landmarker_input_height, crop);
        
        return {crop[0], roi_rect};
    }

    std::pair<std::vector<float>, std::vector<float>> parse_landmarker_outputs(
        const std::vector<Ort::Value> &outputs, 
        const cv::RotatedRect &roi_rect
    )
    {
        assert(outputs.size() == 3);

        const float *confidence_score = outputs[0].GetTensorData<float>();
        const float *handedness_score = outputs[1].GetTensorData<float>();
        const float *landmarks = outputs[2].GetTensorData<float>();

        std::vector<float> scaled_landmarks;
        std::vector<float> normalized_landmarks;
        
        float score = *(confidence_score);
        float handedness = *(handedness_score);

        scaled_landmarks.insert(scaled_landmarks.end(), {score, handedness});
        normalized_landmarks.insert(normalized_landmarks.end(), {score, handedness});

        const float angle = roi_rect.angle * std::numbers::pi / 180.0f;;
        for (size_t i = 0; i < 21; i++)
        {
            float x = *(landmarks + i * 3 + 0);
            float y = *(landmarks + i * 3 + 1);
            float z = *(landmarks + i * 3 + 2);
            
            normalized_landmarks.insert(normalized_landmarks.end(), {x, y, z});

            x -= 0.5f;
            y -= 0.5f;

            float x_orig = x * std::cos(angle) - y * std::sin(angle);
            float y_orig = x * std::sin(angle) + y * std::cos(angle);

            x_orig = x_orig * roi_rect.size.width + roi_rect.center.x;
            y_orig = y_orig * roi_rect.size.height + roi_rect.center.y;

            // x_orig = x_orig * roi_rect.size.width;
            // y_orig = y_orig * roi_rect.size.height;

            scaled_landmarks.insert(scaled_landmarks.end(), {x_orig, y_orig, z});
        }
        return {scaled_landmarks, normalized_landmarks};
    }
}
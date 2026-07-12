/**@file main.cpp
 * @brief Contains code for setting up ONNXsessions, running the models and drawing tracking results
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include "PalmDetector.hpp"
#include "HandLandmarker.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#define NOMINMAX

#include <windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h> 


template <typename T>
size_t vector_product(std::vector<T> &arr)
{
    size_t total = 1;
    size_t arr_size = arr.size();
    for (size_t i = 0; i < arr_size; i++)
    {
        total *= arr[i];
    }
    
    return total;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const ONNXTensorElementDataType &type)
{
    switch (type)
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        os << "undefined";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        os << "float";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        os << "uint8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        os << "int8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        os << "uint16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        os << "int16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        os << "int32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        os << "int64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        os << "std::string";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        os << "bool";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        os << "float16";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        os << "double";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        os << "uint32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        os << "uint64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        os << "float real + float imaginary";
        break;
    case ONNXTensorElementDataType::
        ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        os << "double real + float imaginary";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        os << "bfloat16";
        break;
    default:
        break;
    }

    return os;
}

/**@brief Class for loading, setting up and running ONNX models
 */
class ONNXModel
{   
private:
    Ort::Session &session;
    Ort::AllocatorWithDefaultOptions &allocator;
    Ort::MemoryInfo &mem_info;

    std::vector<Ort::AllocatedStringPtr> name_pointers;
    std::vector<std::vector<int64_t>> inputs_dims, outputs_dims;
    std::vector<const char *> inputs_names, outputs_names;

public:
    ONNXModel(Ort::Session &s, Ort::AllocatorWithDefaultOptions &a, Ort::MemoryInfo &m, const bool verbose = true): 
    session(s), allocator(a), mem_info(m)
    {
        size_t numInputNodes = session.GetInputCount();
        size_t numOutputNodes = session.GetOutputCount();
        if(verbose) std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
        for (size_t i = 0; i < numInputNodes; i++)
        {
            Ort::AllocatedStringPtr inputNamePtr = session.GetInputNameAllocated(i, allocator);
            const char *inputName = inputNamePtr.get();

            Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
            auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

            std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
            
            if(verbose) std::cout << inputName << " -> " << inputType << " " << inputDims << std::endl;

            inputs_dims.push_back(inputDims);
            inputs_names.push_back(inputName);
            name_pointers.push_back(std::move(inputNamePtr));
        }

        if(verbose) std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            Ort::AllocatedStringPtr outputNamePtr = session.GetOutputNameAllocated(i, allocator);
            const char *outputName = outputNamePtr.get();

            Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
            auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

            std::vector<int64_t> outputDims = outputTensorInfo.GetShape();

            if(verbose) std::cout << outputName << " -> " << outputType << " " << outputDims << std::endl;

            outputs_names.push_back(outputName);
            outputs_dims.push_back(outputDims);
            name_pointers.push_back(std::move(outputNamePtr));
        }
    }

    std::vector<Ort::Value> run_inference(const std::vector<cv::Mat> &images)
    {   
        std::vector<Ort::Value> input_tensors;
        std::vector<cv::Mat> img_blobs;
        for (size_t i = 0; i < images.size(); i++)
        {
            cv::Mat img_blob = cv::dnn::blobFromImage(images[i], 1.0/255, cv::Size(inputs_dims[i][2], inputs_dims[i][3]), cv::Scalar(0, 0, 0), true);
            img_blobs.push_back(img_blob);
            
            assert(img_blob.data == img_blobs.back().data);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, 
                (float*)img_blob.data,
                img_blob.total(),
                inputs_dims[i].data(),
                inputs_dims[i].size()
            );

            input_tensors.push_back(std::move(input_tensor));
        }
        
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr}, 
            inputs_names.data(), 
            input_tensors.data(), 
            input_tensors.size(), 
            outputs_names.data(), 
            outputs_names.size()
        );

        return outputs;
    }
    
    std::vector<Ort::Value> run_inference(const cv::Mat &image)
    {   
        std::vector<Ort::Value> input_tensors;
        
        cv::Mat img_blob = cv::dnn::blobFromImage(image, 1.0/255, cv::Size(inputs_dims[0][2], inputs_dims[0][3]), cv::Scalar(0, 0, 0), true);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, 
            (float*)img_blob.data,
            img_blob.total(),
            inputs_dims[0].data(),
            inputs_dims[0].size()
        );

        input_tensors.push_back(std::move(input_tensor));
        
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr}, 
            inputs_names.data(), 
            input_tensors.data(), 
            input_tensors.size(), 
            outputs_names.data(), 
            outputs_names.size()
        );

        return outputs;
    }

    cv::Size get_input_size(const int input_id = 0)
    {
        return cv::Size(inputs_dims[input_id][2], inputs_dims[input_id][3]);
    }
};

/**@brief Helper function to run palm detector inference on Source image
 * @param src_frame Source image
 * @param anchors Flattened vector of anchor boxes
 * @param detector ONNXModel detector model
 * @param landmarker_img_sz Size of detector input frame
 * @param CONF_THRESHOLD Confidence threshold to consider a detection valid
 * @param IOU_THRESHOLD Intersection over Union threshold for NMS
 * @return Pair of vectors, consisting of image crops of hands and their associated rotated Rectangles 
 */
std::pair<std::vector<cv::Mat>, std::vector<cv::RotatedRect>> run_detector(
    cv::Mat &src_frame, 
    const std::vector<double> &anchors, 
    ONNXModel &detector,
    const cv::Size &landmarker_img_sz,
    const float CONF_THRESHOLD,
    const float IOU_THRESHOLD
)
{
    std::vector<cv::Mat> inputs{src_frame};
    std::vector<Ort::Value> detector_outputs = detector.run_inference(inputs);

    std::vector<cv::RotatedRect> crop_rects = PalmDetector::process_detector_outputs(
        detector_outputs, 
        anchors, 
        landmarker_img_sz.width, 
        landmarker_img_sz.height,
        src_frame.cols,
        src_frame.rows,
        src_frame,
        CONF_THRESHOLD, 
        IOU_THRESHOLD
    );
    
    std::vector<cv::Mat> roi_images;
    HandLandmarker::detection_to_roi_crop(
        crop_rects,
        src_frame,
        landmarker_img_sz.width,
        landmarker_img_sz.height,
        roi_images
    );

    return {roi_images, crop_rects};
}

/**@brief Helper function to run landmarker model on detection results
 * @param roi_image RoI image
 * @param roi_rect RoI image's associated rotated rectangles
 * @param landmarker ONNXModel landmarker model to run inference
 * @return Pair of vectors of landmarks.
 *  Both vectors have size of 65 and look like {conf_score, handedness_score, x_1, y_1, z_1, ... , x_21, y_21, z_21}.
 *  .first's xyz of landmarks are in src image coordinates. .second's landmarks are normalized crop image coords.
 */
std::pair<std::vector<float>, std::vector<float>> run_landmarker(
    const cv::Mat &roi_image,
    const cv::RotatedRect &roi_rect,
    ONNXModel &landmarker
)
{
    std::vector<Ort::Value> single_roi_landmarks = landmarker.run_inference(roi_image);

    std::pair<std::vector<float>, std::vector<float>> parsed_landmarks = HandLandmarker::parse_landmarker_outputs(
        single_roi_landmarks, 
        roi_rect
    );

    return parsed_landmarks;
}


/**@brief Calculate and set master volume based on distance between Index and Thumb tips, normalized by RoI width
 * @param src_landmarks Flat vector of landmarks, obtain from running landmarker model
 * @param roi_rect RoI's associated rotated rectangle
 * @param endpointVolume ptr to IAudioEndpointVolume interface
 */
float calculate_and_set_volume(const std::vector<float> &src_landmarks, const cv::RotatedRect &roi_rect, IAudioEndpointVolume *endpointVolume)
{
    const float roi_area = roi_rect.size.width;
    const int ThumbTIP{4};
    const int IndexTIP{8};

    const float dist = std::sqrt(
        std::pow(src_landmarks[2 + 3 * ThumbTIP + 0] - src_landmarks[2 + 3 * IndexTIP + 0], 2) + 
        std::pow(src_landmarks[2 + 3 * ThumbTIP + 1] - src_landmarks[2 + 3 * IndexTIP + 1], 2)
    );

    float volume = (dist / roi_area - 0.05f) / 0.37f;
    volume = std::clamp(volume, 0.0f, 1.0f);

    std::cout << std::format("Area: {}, Distance: {}, Volume: {}", roi_area, dist, volume) << std::endl;
    
    HRESULT hr = endpointVolume->SetMasterVolumeLevelScalar(volume, NULL);  

    return volume;
}

/**@brief Draws Detection results, Landmarks, Image crop, passed to landmarker and the volume bar 
 * @param src_img Source image. Landmarks and connections will be drawn here
 * @param src_landmarks Vector of landmarks in src img coordinates
 * @param normalized landmarks Vector of landmarks in normalized crop img coordinates
 * @param roi_image RoI image
 * @param roi_rect RoI rectangle as cv::RotatedRect
 * @param status_bar_sz Dimensions of volume bar
 * @param is_valid Flag, defining whether the current landmarks are valid.
 */
void display_detection_results(
    cv::Mat &display_window,
    cv::Mat &src_img, 
    const std::vector<float> &src_landmarks,
    const std::vector<float> &normalized_landmarks,
    cv::Mat &roi_image, 
    const cv::RotatedRect &roi_rect,
    const int status_bar_height,
    const bool is_valid_detection,
    const float volume
)
{
    const cv::Size src_sz = src_img.size();
    const cv::Size roi_sz = roi_image.size();

    const float score = src_landmarks[0];
    const float handedness = src_landmarks[1];

    const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4},          // Thumb
        {0, 5}, {5, 6}, {6, 7}, {7, 8},          // Index Finger
        {5, 9}, {9, 10}, {10, 11}, {11, 12},     // Middle Finger
        {9, 13}, {13, 14}, {14, 15}, {15, 16},   // Ring Finger
        {13, 17}, {17, 18}, {18, 19}, {19, 20},  // Pinky
        {0, 17}                                  // Palm base closure
    };

    if (is_valid_detection)
    {
        // Skeleton landmarks are colored based on their z coords
        float z_min{normalized_landmarks[2 + 0 * 3 + 2]}, z_max{normalized_landmarks[2 + 0 * 3 + 2]};
        for (size_t i = 1; i < 21; i++)
        {   
            float z{normalized_landmarks[2 + i * 3 + 2]};
            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
        }
        auto get_z_intensity = [&](int joint_idx) {
            float z = normalized_landmarks[2 + joint_idx * 3 + 2];
            z = (z - z_min) / (z_max - z_min);
            int intensity = (1.0f - z) * 255.0f;
            return std::clamp(intensity, 0, 255);
        };

        // Draw connections
        for (const auto &connection : HAND_CONNECTIONS)
        {
            const int idx1 = connection.first;
            const int idx2 = connection.second;
            
            cv::Point p1(src_landmarks[2 + idx1 * 3 + 0], src_landmarks[2 + idx1 * 3 + 1]);
            cv::Point p2(src_landmarks[2 + idx2 * 3 + 0], src_landmarks[2 + idx2 * 3 + 1]);
            
            // Average the depth intensity of both connected joints for the line color
            int intensity = (get_z_intensity(idx1) + get_z_intensity(idx2)) / 2;
            
            // Draw the skeletal bone line
            cv::line(src_img, p1, p2, cv::Scalar(intensity, intensity, intensity), 2, cv::LINE_AA);
        }

        // Draw Hand Keypoints
        for (size_t i = 0; i < 21; i++)
        {
            float x{src_landmarks[2 + i * 3 + 0]}; 
            float y{src_landmarks[2 + i * 3 + 1]}; 
            
            int intensity = get_z_intensity(i);

            cv::circle(src_img, cv::Point(x, y), 4, cv::Scalar(intensity, intensity, intensity), -1, cv::LINE_AA);
        }

        // Draw rotated bounding box
        cv::Point2f vertices[4];
        roi_rect.points(vertices);
        for (size_t j = 0; j < 4; j++)
        {
            cv::line(src_img, vertices[j], vertices[(j+1)%4], cv::Scalar(255, 255, 255));
        }

        const float font_scale = 0.25f;
        const int font_thickness = 1;
        
        // Draw landmarks on roi crop
        for (size_t i = 0; i < 21; i++)
        {
            float x{normalized_landmarks[2 + i * 3 + 0]}; 
            float y{normalized_landmarks[2 + i * 3 + 1]}; 
            
            int intensity = get_z_intensity(i);

            x *= roi_sz.width;
            y *= roi_sz.height;

            cv::Size text_size = cv::getTextSize(std::to_string(i), cv::HersheyFonts::FONT_ITALIC, font_scale, font_thickness, 0);

            cv::putText(
                roi_image, 
                std::to_string(i), 
                cv::Point(x - 0.5f * text_size.width, y + 0.5f * text_size.height), 
                cv::HersheyFonts::FONT_ITALIC, 
                font_scale, 
                cv::Scalar(0, 0, 0)
            );
        }

        cv::Mat crop_insert_region(display_window, cv::Rect(src_sz.width, 0 + 0 * roi_sz.height, roi_sz.width, roi_sz.height));
        roi_image.copyTo(crop_insert_region);
    }

    cv::Mat frame_insert_region(display_window, cv::Rect(0, 0, src_sz.width, src_sz.height));
    src_img.copyTo(frame_insert_region);


    // Display current master volume using a status bar
    const int margin{10};
    const cv::Point bar_top_left(margin, src_sz.height + margin);
    const cv::Point bar_bot_right(src_sz.width - margin, src_sz.height + status_bar_height - margin);
    const cv::Point bar_filled_bot_right(margin + volume * (src_sz.width - 2 * margin), src_sz.height + status_bar_height - margin);
    
    cv::rectangle(display_window, bar_top_left, bar_bot_right, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(display_window, bar_top_left, bar_bot_right, cv::Scalar(0, 255, 0));
    cv::rectangle(display_window, bar_top_left, bar_filled_bot_right, cv::Scalar(0, 255, 0), cv::FILLED);
}

int main()
{
    const wchar_t *detector_path = L"../../mediapipe_hand-onnx-float/hand_detector.onnx";
    const wchar_t *landmark_path = L"../..//mediapipe_hand-onnx-float/hand_landmark_detector.onnx";
    const std::string palm_anchors_path = "../../mediapipe_hand-onnx-float/anchors_palm.npy";
    
    const float IOU_THRESHOLD = 0.3f;
    const float CONF_THRESHOLD = 0.8f;
    const int status_bar_height = 30;

    std::vector<double> anchors;
    PalmDetector::preload_anchors(palm_anchors_path, anchors);

    // Initialize ONNXruntime env, options and sessions
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HandLandmarkDetector");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // OrtCUDAProviderOptions cuda_opt;
    // cuda_opt.device_id = 0;
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_opt.device_id));

    Ort::Session detector_session(env, detector_path, options);
    Ort::Session landmarker_session(env, landmark_path, options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::cout << "Initializing Palm detector model...\n" << "--------------------------------------" << std::endl;
    ONNXModel palm_detector = ONNXModel(detector_session, allocator, mem_info);
    std::cout << "--------------------------------------" << std::endl;

    std::cout << "Initializing Hand landmarker model...\n" << "--------------------------------------" << std::endl;
    ONNXModel landmarker = ONNXModel(landmarker_session, allocator, mem_info);
    std::cout << "--------------------------------------" << std::endl;

    // Initialize IAudioEndpointVolume to control master volume
    HRESULT hr;  
    
    CoInitialize(NULL);  
    IMMDeviceEnumerator *deviceEnumerator = NULL;  
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), NULL, CLSCTX_INPROC_SERVER, __uuidof(IMMDeviceEnumerator), (LPVOID *)&deviceEnumerator);
    IMMDevice *defaultDevice = NULL;   
    hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &defaultDevice);  
    deviceEnumerator->Release();  deviceEnumerator = NULL;   
    IAudioEndpointVolume *endpointVolume = NULL;  
    hr = defaultDevice->Activate(__uuidof(IAudioEndpointVolume), CLSCTX_INPROC_SERVER, NULL, (LPVOID *)&endpointVolume);  
    defaultDevice->Release();  defaultDevice = NULL;

    float volume;
    hr = endpointVolume->GetMasterVolumeLevelScalar(&volume);  
 
    const cv::Size detector_img_sz = palm_detector.get_input_size();
    const cv::Size landmarker_img_sz = landmarker.get_input_size();

    std::vector<float> last_frame_landmarks;
    bool is_valid_detection = false;

    cv::VideoCapture cap(0);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 480);
    
    cv::Size display_sz(cap.get(cv::CAP_PROP_FRAME_WIDTH) + landmarker_img_sz.width, cap.get(cv::CAP_PROP_FRAME_HEIGHT) + status_bar_height);

    cv::Mat display_window(display_sz, CV_8UC3, cv::Scalar(0, 0, 0));

    while (cap.isOpened())
    {
        cv::Mat src_frame;
        cap.read(src_frame);

        std::pair<std::vector<float>, std::vector<float>> landmarks;
        std::pair<cv::Mat, cv::RotatedRect> roi;
        
        if (is_valid_detection)
        {
            roi = HandLandmarker::landmarks_to_roi_crop(
                last_frame_landmarks, 
                src_frame, 
                landmarker_img_sz.width, 
                landmarker_img_sz.height
            );

            landmarks = run_landmarker(roi.first, roi.second, landmarker);

            const float score = landmarks.first[0];
            const float handedness = landmarks.first[1];

            if (score < CONF_THRESHOLD) 
            {
                is_valid_detection = false;
            }
            else
            {
                last_frame_landmarks = landmarks.first;
                volume = calculate_and_set_volume(landmarks.first, roi.second, endpointVolume);
                is_valid_detection = true;
            }
        }

        if (!is_valid_detection)
        {
            std::pair<std::vector<cv::Mat>, std::vector<cv::RotatedRect>> rois = run_detector(
                src_frame, 
                anchors, 
                palm_detector, 
                landmarker_img_sz, 
                CONF_THRESHOLD, 
                IOU_THRESHOLD
            );

            std::vector<std::vector<float>> src_landmarks, normalized_landmarks;

            for (size_t i = 0; i < std::min((int)rois.first.size(), 1); i++)
            {   
                std::pair<std::vector<float>, std::vector<float>> parsed_landmarks = run_landmarker(rois.first[i], rois.second[i], landmarker);

                src_landmarks.push_back(parsed_landmarks.first);
                normalized_landmarks.push_back(parsed_landmarks.second);
            }

            int hand_id = src_landmarks.empty() ? -1: 0;

            if (hand_id != -1)
            {
                landmarks = {src_landmarks[hand_id], normalized_landmarks[hand_id]};
                roi = {rois.first[hand_id], rois.second[hand_id]};
                last_frame_landmarks = landmarks.first;
                volume = calculate_and_set_volume(landmarks.first, roi.second, endpointVolume);

                is_valid_detection = true;
                
                // HandLandmarker::landmarks_to_roi_crop(
                //     landmarks.first, 
                //     src_frame, 
                //     landmarker_img_sz.width, 
                //     landmarker_img_sz.height
                // );
            }
            else
            {
                is_valid_detection = false;
            }
        }

        display_detection_results(
            display_window,
            src_frame, 
            landmarks.first, 
            landmarks.second, 
            roi.first, 
            roi.second,
            status_bar_height,
            is_valid_detection,
            volume
        );
        
        
        cv::imshow("Cam", display_window);
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    endpointVolume->Release();   
    CoUninitialize();  
    
    return 0;
}
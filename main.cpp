#include "include/inference_engine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

// Function to simulate loading a model (e.g., from a file)
std::string get_model_path(const std::string& model_name) {
    // In a real application, this would load from a specific path or registry
    if (model_name == "image_classifier") {
        return "models/image_classifier.onnx";
    } else if (model_name == "object_detector") {
        return "models/object_detector.pb"; // Example for TensorFlow frozen graph
    } else {
        return "";
    }
}

int main() {
    std::cout << "AI Inference Engine Demo" << std::endl;

    InferenceEngine engine;

    // --- Test Case 1: Image Classification Model ---
    std::cout << "\n--- Testing Image Classification Model ---" << std::endl;
    std::string image_model_path = get_model_path("image_classifier");
    if (engine.loadModel(image_model_path, 1 * 3 * 224 * 224, 1000)) { // Batch 1, 3 channels, 224x224, 1000 classes
        std::vector<float> input_data(1 * 3 * 224 * 224, 0.1f); // Dummy input
        std::vector<float> output_data;

        auto start = std::chrono::high_resolution_clock::now();
        if (engine.runInference(input_data, output_data)) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Inference successful for image classifier. Output size: " << output_data.size() 
                      << ", took " << duration.count() * 1000 << " ms." << std::endl;
            // Simulate post-processing: find top 3 classes
            std::vector<std::pair<float, int>> predictions;
            for (int i = 0; i < output_data.size(); ++i) {
                predictions.push_back({output_data[i], i});
            }
            std::sort(predictions.rbegin(), predictions.rend());
            std::cout << "Top 3 predictions:" << std::endl;
            for (int i = 0; i < std::min((int)predictions.size(), 3); ++i) {
                std::cout << "  Class " << predictions[i].second << ": " << predictions[i].first << std::endl;
            }
        } else {
            std::cerr << "Inference failed for image classifier!" << std::endl;
        }
    }

    // --- Test Case 2: Object Detection Model ---
    std::cout << "\n--- Testing Object Detection Model ---" << std::endl;
    std::string object_model_path = get_model_path("object_detector");
    if (engine.loadModel(object_model_path, 1 * 3 * 640 * 640, 7)) { // Example: YOLO-like output (boxes, scores, classes)
        std::vector<float> input_data(1 * 3 * 640 * 640, 0.5f); // Dummy input
        std::vector<float> output_data;

        auto start = std::chrono::high_resolution_clock::now();
        if (engine.runInference(input_data, output_data)) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Inference successful for object detector. Output size: " << output_data.size() 
                      << ", took " << duration.count() * 1000 << " ms." << std::endl;
            // Simulate parsing object detection output
            if (output_data.size() > 0) {
                std::cout << "Detected objects (simulated):" << std::endl;
                // In a real scenario, parse bounding boxes, scores, and classes
                std::cout << "  Object 1: [x, y, w, h, score, class_id]" << std::endl;
            }
        } else {
            std::cerr << "Inference failed for object detector!" << std::endl;
        }
    }

    std::cout << "\nAI Inference Engine Demo Finished." << std::endl;
    return 0;
}
// Update on 2023-07-10 00:00:00 - 350

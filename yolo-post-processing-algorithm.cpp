#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

struct Detection {
    float x1, y1, x2, y2; 
    float confidence;
    int label; 
};
float calculate_iou(const Detection& a, const Detection& b) {
    float x_overlap = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    float y_overlap = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float intersection = x_overlap * y_overlap;
    
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    return (union_area > 0) ? (intersection / union_area) : 0;
}
std::vector<Detection> apply_nms(std::vector<Detection>& candidates, float iou_thresh) {
    std::vector<Detection> results;
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> discarded(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (discarded[i]) continue;
        results.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!discarded[j] && calculate_iou(candidates[i], candidates[j]) > iou_thresh) {
                discarded[j] = true; 
            }
        }
    }
    return results;
}
int main() {
    std::vector<Detection> raw_neural_output = {
        {150, 150, 250, 250, 0.98, 1}, 
        {155, 148, 255, 248, 0.85, 1}, 
        {145, 152, 245, 252, 0.72, 1}, 
        {600, 100, 700, 200, 0.91, 1}  
    };
    std::cout << "--- FRACTURE DETECTION ALGORITHM ---" << std::endl;
    std::cout << "Raw Output Count: " << raw_neural_output.size() << std::endl;
    std::vector<Detection> final_detections = apply_nms(raw_neural_output, 0.45);
    std::cout << "Final Medical Report: " << final_detections.size() << " Fracture(s) found.\n" << std::endl;
    for (const auto& d : final_detections) {
        std::cout << "Fracture Location: [" << d.x1 << ", " << d.y1 << "]" 
                  << " Confidence: " << (d.confidence * 100) << "%" << std::endl;
    }
    return 0;
}

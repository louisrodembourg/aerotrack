// TensorRT runtime: an engine built by `aerotrack.inference.export_engine`, run from C++.
//
// The three stages match the Python Detector's, so their timings compare one to one:
//   preprocess   letterbox on the host into pinned memory, upload the 8-bit image, then
//                BGR->RGB, HWC->CHW and /255 in a CUDA kernel (Python: cv2 letterbox,
//                upload, conversion on the GPU)
//   inference    enqueueV3
//   postprocess  download the raw head output, NMS and rescaling on the host (Python:
//                the same, host_nms)
// Each stage synchronises its stream before returning.
#pragma once

#include <cuda_runtime_api.h>

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

namespace aerotrack {

class TrtDetector {
 public:
  TrtDetector(const std::string& engine_path, int imgsz, float conf, double iou, int max_det,
              int device);
  ~TrtDetector();
  TrtDetector(const TrtDetector&) = delete;
  TrtDetector& operator=(const TrtDetector&) = delete;

  void preprocess(const uint8_t* bgr, int h, int w);
  void infer();
  std::vector<float> postprocess();

  int channels() const { return channels_; }
  int anchors() const { return anchors_; }
  std::string input_dtype() const;
  std::string output_dtype() const;

 private:
  struct Deleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
  };

  int imgsz_, max_det_, device_;
  float conf_;
  double iou_;
  int h0_ = 0, w0_ = 0;  // original size of the frame being processed
  int channels_ = 0, anchors_ = 0;
  nvinfer1::DataType in_type_{}, out_type_{};
  std::string in_name_, out_name_;

  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  cudaStream_t stream_ = nullptr;

  uint8_t* host_img_ = nullptr;   // pinned letterboxed frame
  uint8_t* dev_img_ = nullptr;
  void* dev_in_ = nullptr;        // network input (float or half)
  void* dev_out_ = nullptr;       // raw head output
  void* host_out_ = nullptr;      // pinned copy of it
  std::vector<float> out_float_;  // the output as float, for a half engine
};

// Implemented in preprocess.cu.
void launch_to_chw(const uint8_t* bgr, void* out, bool half, int size, cudaStream_t stream);

}  // namespace aerotrack

// TensorRT runtime: an engine built by `aerotrack.inference.export_engine`, run from C++.
//
// The three stages match the Python Detector's, so their timings compare one to one:
//   preprocess   letterbox on the host into pinned memory, upload the 8-bit image, then
//                BGR->RGB, HWC->CHW and /255 in a CUDA kernel (Python: cv2 letterbox,
//                upload, conversion on the GPU); or, with gpu_preprocess, upload the
//                original frame and letterbox it on the GPU too, same pixels
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

#include "pipeline.hpp"

namespace aerotrack {

// ResizeTaps on the device (raw pointers, passed to the kernel by value).
struct DeviceTaps {
  const int *x0, *x1, *y0, *y1;
  const int16_t *ax0, *ax1, *by0, *by1;
};

class TrtDetector {
 public:
  TrtDetector(const std::string& engine_path, int imgsz, float conf, double iou, int max_det,
              int device, bool gpu_preprocess);
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

  void upload_taps(int h, int w);  // when the frame size changes

  int imgsz_, max_det_, device_;
  bool gpu_pre_;
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

  // gpu_preprocess: the original frame (pinned + device, grown as needed) and the resize
  // taps of the current frame size, in one device allocation.
  uint8_t* host_src_ = nullptr;
  uint8_t* dev_src_ = nullptr;
  size_t src_capacity_ = 0;
  void* dev_taps_ = nullptr;
  DeviceTaps taps_{};
  Letterbox lb_{};
  int taps_h_ = 0, taps_w_ = 0;
};

// Implemented in preprocess.cu.
void launch_to_chw(const uint8_t* bgr, void* out, bool half, int size, cudaStream_t stream);
void launch_letterbox_chw(const uint8_t* src, int sw, const DeviceTaps& taps,
                          const Letterbox& lb, void* out, bool half, int size,
                          cudaStream_t stream);

}  // namespace aerotrack

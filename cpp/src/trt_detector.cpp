#include "trt_detector.hpp"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "pipeline.hpp"

namespace aerotrack {

namespace {

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << '\n';
  }
};

Logger& logger() {
  static Logger instance;  // must outlive every runtime
  return instance;
}

size_t element_size(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    default: throw std::runtime_error("engine I/O must be float32 or float16");
  }
}

std::string dtype_name(nvinfer1::DataType t) {
  return t == nvinfer1::DataType::kHALF ? "float16" : "float32";
}

// The engine file as Ultralytics writes it: a little-endian int32 length, that many bytes
// of JSON metadata, then the serialized engine. A bare engine has no such header.
std::vector<char> read_engine(const std::string& path, size_t& offset) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open engine " + path);
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  offset = 0;
  if (data.size() > 4) {
    int32_t len;
    std::memcpy(&len, data.data(), 4);
    if (len > 0 && static_cast<size_t>(len) + 4 < data.size() && data[4] == '{') offset = 4 + len;
  }
  return data;
}

}  // namespace

TrtDetector::TrtDetector(const std::string& engine_path, int imgsz, float conf, double iou,
                         int max_det, int device)
    : imgsz_(imgsz), max_det_(max_det), device_(device), conf_(conf), iou_(iou) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  size_t offset;
  std::vector<char> blob = read_engine(engine_path, offset);
  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_.reset(runtime_->deserializeCudaEngine(blob.data() + offset, blob.size() - offset));
  if (!engine_) throw std::runtime_error("cannot deserialize " + engine_path);
  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("createExecutionContext failed");

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getTensorName(i);
    nvinfer1::Dims d = engine_->getTensorShape(name);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      if (!in_name_.empty()) throw std::runtime_error("engine has more than one input");
      if (d.nbDims != 4 || d.d[0] != 1 || d.d[1] != 3 || d.d[2] != imgsz || d.d[3] != imgsz)
        throw std::runtime_error("engine input is not 1 x 3 x imgsz x imgsz");
      in_name_ = name;
      in_type_ = engine_->getTensorDataType(name);
    } else {
      if (!out_name_.empty()) throw std::runtime_error("engine has more than one output");
      // One-to-many head: 1 x (4 + nc) x anchors. An NMS-free export (1 x max_det x 6)
      // would need another postprocess; every backend runs the one-to-many head.
      if (d.nbDims != 3 || d.d[0] != 1 || d.d[1] < 5 || d.d[2] == 6)
        throw std::runtime_error("engine output is not 1 x (4 + nc) x anchors");
      out_name_ = name;
      out_type_ = engine_->getTensorDataType(name);
      channels_ = static_cast<int>(d.d[1]);
      anchors_ = static_cast<int>(d.d[2]);
    }
  }
  if (in_name_.empty() || out_name_.empty()) throw std::runtime_error("engine I/O not found");

  const size_t img_bytes = static_cast<size_t>(imgsz) * imgsz * 3;
  const size_t out_bytes = static_cast<size_t>(channels_) * anchors_ * element_size(out_type_);
  check(cudaStreamCreate(&stream_), "cudaStreamCreate");
  check(cudaMallocHost(&host_img_, img_bytes), "cudaMallocHost");
  check(cudaMalloc(&dev_img_, img_bytes), "cudaMalloc");
  check(cudaMalloc(&dev_in_, img_bytes * element_size(in_type_)), "cudaMalloc");
  check(cudaMalloc(&dev_out_, out_bytes), "cudaMalloc");
  check(cudaMallocHost(&host_out_, out_bytes), "cudaMallocHost");
  if (out_type_ == nvinfer1::DataType::kHALF)
    out_float_.resize(static_cast<size_t>(channels_) * anchors_);
  if (!context_->setTensorAddress(in_name_.c_str(), dev_in_) ||
      !context_->setTensorAddress(out_name_.c_str(), dev_out_))
    throw std::runtime_error("setTensorAddress failed");
}

TrtDetector::~TrtDetector() {
  context_.reset();  // before the buffers it points to
  cudaFreeHost(host_out_);
  cudaFree(dev_out_);
  cudaFree(dev_in_);
  cudaFree(dev_img_);
  cudaFreeHost(host_img_);
  if (stream_) cudaStreamDestroy(stream_);
}

std::string TrtDetector::input_dtype() const { return dtype_name(in_type_); }
std::string TrtDetector::output_dtype() const { return dtype_name(out_type_); }

void TrtDetector::preprocess(const uint8_t* bgr, int h, int w) {
  h0_ = h, w0_ = w;
  letterbox(bgr, h, w, imgsz_, host_img_);
  check(cudaMemcpyAsync(dev_img_, host_img_, static_cast<size_t>(imgsz_) * imgsz_ * 3,
                        cudaMemcpyHostToDevice, stream_), "upload");
  launch_to_chw(dev_img_, dev_in_, in_type_ == nvinfer1::DataType::kHALF, imgsz_, stream_);
  check(cudaGetLastError(), "to_chw kernel");
  check(cudaStreamSynchronize(stream_), "preprocess");
}

void TrtDetector::infer() {
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  check(cudaStreamSynchronize(stream_), "inference");
}

std::vector<float> TrtDetector::postprocess() {
  const size_t count = static_cast<size_t>(channels_) * anchors_;
  check(cudaMemcpyAsync(host_out_, dev_out_, count * element_size(out_type_),
                        cudaMemcpyDeviceToHost, stream_), "download");
  check(cudaStreamSynchronize(stream_), "postprocess");
  const float* raw = static_cast<const float*>(host_out_);
  if (out_type_ == nvinfer1::DataType::kHALF) {
    const __half* h = static_cast<const __half*>(host_out_);
    for (size_t i = 0; i < count; ++i) out_float_[i] = __half2float(h[i]);
    raw = out_float_.data();
  }
  return aerotrack::postprocess(raw, channels_, anchors_, imgsz_, h0_, w0_, conf_, iou_,
                                max_det_);
}

}  // namespace aerotrack

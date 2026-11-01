// Letterboxed 8-bit BGR (HWC) -> network input RGB (CHW) / 255, on the GPU: the same
// conversion the Python Detector runs on the device after uploading the 8-bit image.
#include <cuda_fp16.h>

#include "trt_detector.hpp"

namespace aerotrack {

namespace {

template <typename T>
__device__ T convert(float v);
template <>
__device__ float convert<float>(float v) { return v; }
template <>
__device__ __half convert<__half>(float v) { return __float2half(v); }

template <typename T>
__global__ void to_chw(const uint8_t* __restrict__ bgr, T* __restrict__ out, int plane) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= plane) return;
  const uint8_t* px = bgr + 3 * i;
  out[i] = convert<T>(px[2] / 255.f);  // R
  out[plane + i] = convert<T>(px[1] / 255.f);
  out[2 * plane + i] = convert<T>(px[0] / 255.f);
}

}  // namespace

void launch_to_chw(const uint8_t* bgr, void* out, bool half, int size, cudaStream_t stream) {
  int plane = size * size, threads = 256, blocks = (plane + threads - 1) / threads;
  if (half)
    to_chw<<<blocks, threads, 0, stream>>>(bgr, static_cast<__half*>(out), plane);
  else
    to_chw<<<blocks, threads, 0, stream>>>(bgr, static_cast<float*>(out), plane);
}

}  // namespace aerotrack

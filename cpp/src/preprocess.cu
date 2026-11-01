// Network input on the GPU. Two kernels:
//   to_chw          letterboxed 8-bit BGR (HWC) -> RGB (CHW) / 255: the conversion the
//                   Python Detector runs on the device after uploading the 8-bit image
//   letterbox_chw   the same from the ORIGINAL frame: resize (the host's taps and integer
//                   arithmetic, so the same pixels), padding and conversion in one pass
#include <cuda_fp16.h>

#include "pipeline.hpp"
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

template <typename T>
__global__ void letterbox_chw(const uint8_t* __restrict__ src, int sw, DeviceTaps t,
                              Letterbox lb, int size, T* __restrict__ out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size || y >= size) return;
  int dx = x - lb.left, dy = y - lb.top;
  int v[3] = {114, 114, 114};  // Ultralytics' padding value
  if (dx >= 0 && dx < lb.new_w && dy >= 0 && dy < lb.new_h) {
    const uint8_t* r0 = src + t.y0[dy] * sw * 3;
    const uint8_t* r1 = src + t.y1[dy] * sw * 3;
    int c0 = t.x0[dx] * 3, c1 = t.x1[dx] * 3, a0 = t.ax0[dx], a1 = t.ax1[dx];
    for (int c = 0; c < 3; ++c) {
      int h0 = r0[c0 + c] * a0 + r0[c1 + c] * a1;
      int h1 = r1[c0 + c] * a0 + r1[c1 + c] * a1;
      v[c] = resize_round(h0 * t.by0[dy] + h1 * t.by1[dy]);
    }
  }
  int plane = size * size, i = y * size + x;
  out[i] = convert<T>(v[2] / 255.f);  // R
  out[plane + i] = convert<T>(v[1] / 255.f);
  out[2 * plane + i] = convert<T>(v[0] / 255.f);
}

}  // namespace

void launch_letterbox_chw(const uint8_t* src, int sw, const DeviceTaps& taps,
                          const Letterbox& lb, void* out, bool half, int size,
                          cudaStream_t stream) {
  dim3 threads(32, 8), blocks((size + 31) / 32, (size + 7) / 8);
  if (half)
    letterbox_chw<<<blocks, threads, 0, stream>>>(src, sw, taps, lb, size,
                                                  static_cast<__half*>(out));
  else
    letterbox_chw<<<blocks, threads, 0, stream>>>(src, sw, taps, lb, size,
                                                  static_cast<float*>(out));
}

void launch_to_chw(const uint8_t* bgr, void* out, bool half, int size, cudaStream_t stream) {
  int plane = size * size, threads = 256, blocks = (plane + threads - 1) / threads;
  if (half)
    to_chw<<<blocks, threads, 0, stream>>>(bgr, static_cast<__half*>(out), plane);
  else
    to_chw<<<blocks, threads, 0, stream>>>(bgr, static_cast<float*>(out), plane);
}

}  // namespace aerotrack

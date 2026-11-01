// Host-side stages of the C++ runtime: letterbox and NMS + rescaling.
//
// Both reproduce what the Python Detector does (Ultralytics LetterBox, cv2.resize
// INTER_LINEAR, Ultralytics non_max_suppression with torchvision NMS, scale_boxes) so the
// C++ and Python pipelines can be compared stage by stage. No OpenCV, no CUDA: this file
// builds on any machine and is tested against the Python stages (tests/test_cpp.py).
#pragma once

#include <cstdint>
#include <vector>

namespace aerotrack {

struct Letterbox {
  int new_w, new_h;  // resized image, before padding
  int left, top;     // padding before the image
};

// Ultralytics LetterBox(new_shape=(size, size), auto=False, center=True, scaleup=True).
Letterbox letterbox_params(int h, int w, int size);

// cv2.resize(src, (dw, dh), interpolation=INTER_LINEAR_EXACT) for 3-channel 8-bit images:
// bilinear, 11-bit fixed-point weights, bit-exact on every platform. Ultralytics' LetterBox
// uses INTER_LINEAR, whose rounding depends on the build (SIMD path, vendor HAL): on an
// M3 Pro with OpenCV 5.0 it differs from INTER_LINEAR_EXACT by 1 on ~20 % of the values,
// never more (DECISIONS.md). The exact variant is the reproducible one.
void resize_linear(const uint8_t* src, int sh, int sw, uint8_t* dst, int dh, int dw);

// Letterboxed BGR image (size x size x 3, padding 114) written to `dst`.
void letterbox(const uint8_t* src, int h, int w, int size, uint8_t* dst);

// Ultralytics non_max_suppression (one-to-many head, best class per anchor, class-aware
// torchvision NMS) then scale_boxes to the original h0 x w0 image.
// `raw` is the head output, channel-major: (4 + nc) x n, rows cx, cy, w, h, class scores.
// Returns rows of x1, y1, x2, y2, conf, class.
std::vector<float> postprocess(const float* raw, int channels, int n, int size, int h0, int w0,
                               float conf, double iou, int max_det);

}  // namespace aerotrack

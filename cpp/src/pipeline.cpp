#include "pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace aerotrack {

namespace {

constexpr float kMaxWh = 7680.f;  // Ultralytics: class offset for class-aware NMS
constexpr int kMaxNms = 30000;    // Ultralytics: candidates kept before NMS
constexpr uint8_t kPad = 114;

// Python's round(): half to even, as std::nearbyint in the default rounding mode.
int py_round(double v) { return static_cast<int>(std::nearbyint(v)); }

int16_t coef(float v) {  // saturate_cast<short>(v * INTER_RESIZE_COEF_SCALE)
  long r = std::lrint(v * kCoefScale);
  return static_cast<int16_t>(std::clamp<long>(r, INT16_MIN, INT16_MAX));
}

}  // namespace

Letterbox letterbox_params(int h, int w, int size) {
  double r = std::min(static_cast<double>(size) / h, static_cast<double>(size) / w);
  Letterbox lb{};
  lb.new_w = std::max(py_round(w * r), 1);
  lb.new_h = std::max(py_round(h * r), 1);
  lb.left = py_round((size - lb.new_w) / 2.0 - 0.1);
  lb.top = py_round((size - lb.new_h) / 2.0 - 0.1);
  return lb;
}

// Same taps as OpenCV's resizeGeneric. A column past the last one is clamped with a zero
// weight, which reproduces OpenCV's own border handling (it then uses the last column alone).
ResizeTaps resize_taps(int sh, int sw, int dh, int dw) {
  // cv::resize: inv_scale = dsize / ssize, scale = 1 / inv_scale.
  double scale_x = 1. / (static_cast<double>(dw) / sw);
  double scale_y = 1. / (static_cast<double>(dh) / sh);
  ResizeTaps t;
  t.x0.resize(dw), t.x1.resize(dw), t.ax0.resize(dw), t.ax1.resize(dw);
  for (int dx = 0; dx < dw; ++dx) {
    float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    int sx = static_cast<int>(std::floor(fx));
    fx -= sx;
    if (sx < 0) fx = 0, sx = 0;
    if (sx >= sw - 1) fx = 0, sx = sw - 1;
    t.x0[dx] = sx, t.x1[dx] = std::min(sx + 1, sw - 1);
    t.ax0[dx] = coef(1.f - fx), t.ax1[dx] = coef(fx);
  }
  t.y0.resize(dh), t.y1.resize(dh), t.by0.resize(dh), t.by1.resize(dh);
  for (int dy = 0; dy < dh; ++dy) {
    float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    int sy = static_cast<int>(std::floor(fy));
    fy -= sy;
    // Rows are clamped to the image; the weights are not (OpenCV's resizeGeneric).
    t.y0[dy] = std::clamp(sy, 0, sh - 1), t.y1[dy] = std::clamp(sy + 1, 0, sh - 1);
    t.by0[dy] = coef(1.f - fy), t.by1[dy] = coef(fy);
  }
  return t;
}

void resize_linear(const uint8_t* src, int sh, int sw, uint8_t* dst, int dh, int dw) {
  constexpr int cn = 3;
  const ResizeTaps t = resize_taps(sh, sw, dh, dw);

  // Each horizontally resized source row is computed once and kept while rows need it.
  std::vector<int> rows(2 * dw * cn);
  int cached[2] = {-1, -1};
  auto hrow = [&](int sy, int slot) {
    if (cached[slot] == sy) return;
    const uint8_t* s = src + sy * sw * cn;
    int* d = rows.data() + slot * dw * cn;
    for (int dx = 0; dx < dw; ++dx)
      for (int c = 0; c < cn; ++c)
        d[dx * cn + c] = s[t.x0[dx] * cn + c] * t.ax0[dx] + s[t.x1[dx] * cn + c] * t.ax1[dx];
    cached[slot] = sy;
  };

  for (int dy = 0; dy < dh; ++dy) {
    int y0 = t.y0[dy], y1 = t.y1[dy];
    // Two slots; swap rather than recompute when the previous bottom row becomes the top.
    if (cached[1] == y0 && cached[0] != y0) {
      std::swap_ranges(rows.begin(), rows.begin() + dw * cn, rows.begin() + dw * cn);
      std::swap(cached[0], cached[1]);
    }
    hrow(y0, 0);
    hrow(y1, 1);
    const int* s0 = rows.data();
    const int* s1 = rows.data() + dw * cn;
    uint8_t* d = dst + dy * dw * cn;
    for (int x = 0; x < dw * cn; ++x)
      d[x] = static_cast<uint8_t>(resize_round(s0[x] * t.by0[dy] + s1[x] * t.by1[dy]));
  }
}

void letterbox(const uint8_t* src, int h, int w, int size, uint8_t* dst) {
  constexpr int cn = 3;
  Letterbox lb = letterbox_params(h, w, size);
  std::memset(dst, kPad, static_cast<size_t>(size) * size * cn);
  std::vector<uint8_t> resized;
  const uint8_t* img = src;
  if (lb.new_w != w || lb.new_h != h) {
    resized.resize(static_cast<size_t>(lb.new_w) * lb.new_h * cn);
    resize_linear(src, h, w, resized.data(), lb.new_h, lb.new_w);
    img = resized.data();
  }
  for (int y = 0; y < lb.new_h; ++y)
    std::memcpy(dst + ((y + lb.top) * size + lb.left) * cn, img + y * lb.new_w * cn,
                static_cast<size_t>(lb.new_w) * cn);
}

std::vector<float> postprocess(const float* raw, int channels, int n, int size, int h0, int w0,
                               float conf, double iou, int max_det) {
  const int nc = channels - 4;
  struct Cand {
    float x1, y1, x2, y2, score, cls;
  };
  std::vector<Cand> cand;
  for (int i = 0; i < n; ++i) {
    int best = 0;
    float score = raw[4 * n + i];
    for (int c = 1; c < nc; ++c) {  // first maximum, as torch.max
      float v = raw[(4 + c) * n + i];
      if (v > score) score = v, best = c;
    }
    if (!(score > conf)) continue;
    float cx = raw[i], cy = raw[n + i], hw = raw[2 * n + i] / 2, hh = raw[3 * n + i] / 2;
    cand.push_back({cx - hw, cy - hh, cx + hw, cy + hh, score, static_cast<float>(best)});
  }
  // Candidates by descending score; Ultralytics first caps their number.
  std::vector<int> order(cand.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return cand[a].score > cand[b].score; });
  if (order.size() > kMaxNms) order.resize(kMaxNms);

  // torchvision.ops.nms on boxes offset by class (class-aware), float arithmetic.
  const size_t m = order.size();
  std::vector<float> bx1(m), by1(m), bx2(m), by2(m), area(m);
  for (size_t k = 0; k < m; ++k) {
    const Cand& c = cand[order[k]];
    float off = c.cls * kMaxWh;
    bx1[k] = c.x1 + off, by1[k] = c.y1 + off, bx2[k] = c.x2 + off, by2[k] = c.y2 + off;
    area[k] = (bx2[k] - bx1[k]) * (by2[k] - by1[k]);
  }
  std::vector<char> suppressed(m, 0);
  std::vector<int> keep;
  for (size_t i = 0; i < m && static_cast<int>(keep.size()) < max_det; ++i) {
    if (suppressed[i]) continue;
    keep.push_back(static_cast<int>(i));
    for (size_t j = i + 1; j < m; ++j) {
      if (suppressed[j]) continue;
      float w = std::max(0.f, std::min(bx2[i], bx2[j]) - std::max(bx1[i], bx1[j]));
      float h = std::max(0.f, std::min(by2[i], by2[j]) - std::max(by1[i], by1[j]));
      float inter = w * h;
      float ovr = inter / (area[i] + area[j] - inter);
      if (ovr > iou) suppressed[j] = 1;
    }
  }

  // scale_boxes: remove the letterbox padding, undo the per-axis gain, clip.
  Letterbox lb = letterbox_params(h0, w0, size);
  const float gx = static_cast<float>(static_cast<double>(lb.new_w) / w0);
  const float gy = static_cast<float>(static_cast<double>(lb.new_h) / h0);
  const float px = static_cast<float>(lb.left), py = static_cast<float>(lb.top);
  std::vector<float> out;
  out.reserve(keep.size() * 6);
  for (int k : keep) {
    const Cand& c = cand[order[k]];
    out.push_back(std::clamp((c.x1 - px) / gx, 0.f, static_cast<float>(w0)));
    out.push_back(std::clamp((c.y1 - py) / gy, 0.f, static_cast<float>(h0)));
    out.push_back(std::clamp((c.x2 - px) / gx, 0.f, static_cast<float>(w0)));
    out.push_back(std::clamp((c.y2 - py) / gy, 0.f, static_cast<float>(h0)));
    out.push_back(c.score);
    out.push_back(c.cls);
  }
  return out;
}

}  // namespace aerotrack

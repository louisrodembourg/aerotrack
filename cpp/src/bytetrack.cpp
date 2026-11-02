#include "bytetrack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace aerotrack {

namespace {

constexpr int kTracked = 1, kLost = 2, kRemoved = 3;
constexpr double kStdPos = 1.0 / 20, kStdVel = 1.0 / 160;  // kalman.py

// ---- Kalman filter (aerotrack/track/kalman.py) -----------------------------------------

void xyxy_to_xyah(const double* b, double z[4]) {
  double w = b[2] - b[0], h = b[3] - b[1];
  z[0] = b[0] + w / 2, z[1] = b[1] + h / 2, z[2] = w / h, z[3] = h;
}

// Per-dimension standard deviations scaled by the box height h (kalman.py `_std`).
void stds(double h, double pos, double vel, double a_pos, double a_vel, double s[8]) {
  s[0] = pos * h, s[1] = pos * h, s[2] = a_pos, s[3] = pos * h;
  s[4] = vel * h, s[5] = vel * h, s[6] = a_vel, s[7] = vel * h;
}

void kf_initiate(const double z[4], double mean[8], double cov[64]) {
  for (int i = 0; i < 4; ++i) mean[i] = z[i], mean[i + 4] = 0;
  double s[8];
  stds(z[3], 2 * kStdPos, 10 * kStdVel, 1e-2, 1e-5, s);
  std::fill(cov, cov + 64, 0.0);
  for (int i = 0; i < 8; ++i) cov[i * 9] = s[i] * s[i];
}

void kf_predict(double mean[8], double cov[64]) {
  double s[8];
  stds(mean[3], kStdPos, kStdVel, 1e-2, 1e-5, s);  // from the mean before the step
  for (int i = 0; i < 4; ++i) mean[i] = mean[i] + mean[i + 4];  // F has dt = 1
  double a[64];  // F P
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) a[i * 8 + j] = i < 4 ? cov[i * 8 + j] + cov[(i + 4) * 8 + j]
                                                     : cov[i * 8 + j];
  for (int i = 0; i < 8; ++i)  // (F P) F^T + Q
    for (int j = 0; j < 8; ++j)
      cov[i * 8 + j] = (j < 4 ? a[i * 8 + j] + a[i * 8 + j + 4] : a[i * 8 + j]) +
                       (i == j ? s[i] * s[i] : 0.0);
}

// Solve S X = B for X (S 4 x 4, B 4 x 8), LU with partial pivoting as LAPACK's gesv.
void solve4(double S[16], double B[32]) {
  for (int k = 0; k < 4; ++k) {
    int p = k;
    for (int i = k + 1; i < 4; ++i)
      if (std::abs(S[i * 4 + k]) > std::abs(S[p * 4 + k])) p = i;
    if (p != k) {
      for (int j = 0; j < 4; ++j) std::swap(S[k * 4 + j], S[p * 4 + j]);
      for (int j = 0; j < 8; ++j) std::swap(B[k * 8 + j], B[p * 8 + j]);
    }
    for (int i = k + 1; i < 4; ++i) {
      double f = S[i * 4 + k] / S[k * 4 + k];
      for (int j = k; j < 4; ++j) S[i * 4 + j] -= f * S[k * 4 + j];
      for (int j = 0; j < 8; ++j) B[i * 8 + j] -= f * B[k * 8 + j];
    }
  }
  for (int k = 3; k >= 0; --k)
    for (int j = 0; j < 8; ++j) {
      double v = B[k * 8 + j];
      for (int i = k + 1; i < 4; ++i) v -= S[k * 4 + i] * B[i * 8 + j];
      B[k * 8 + j] = v / S[k * 4 + k];
    }
}

void kf_update(double mean[8], double cov[64], const double z[4]) {
  double s[8];
  stds(mean[3], kStdPos, 0, 1e-1, 0, s);
  double S[16], S0[16], B[32];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) S[i * 4 + j] = cov[i * 8 + j] + (i == j ? s[i] * s[i] : 0.0);
  std::copy(S, S + 16, S0);
  for (int i = 0; i < 4; ++i)  // (P H^T)^T
    for (int j = 0; j < 8; ++j) B[i * 8 + j] = cov[j * 8 + i];
  solve4(S, B);  // B <- K^T
  double innov[4];
  for (int i = 0; i < 4; ++i) innov[i] = z[i] - mean[i];
  for (int i = 0; i < 8; ++i) {
    double v = 0;
    for (int k = 0; k < 4; ++k) v += B[k * 8 + i] * innov[k];
    mean[i] += v;
  }
  double KS[32];  // K S, 8 x 4
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) {
      double v = 0;
      for (int k = 0; k < 4; ++k) v += B[k * 8 + i] * S0[k * 4 + j];
      KS[i * 4 + j] = v;
    }
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      double v = 0;
      for (int k = 0; k < 4; ++k) v += KS[i * 4 + k] * B[k * 8 + j];
      cov[i * 8 + j] -= v;
    }
}

double iou(const double* a, const double* b) {
  double w = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  double h = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  double inter = w * h;
  double area_a = (a[2] - a[0]) * (a[3] - a[1]), area_b = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (area_a + area_b - inter + 1e-7);
}

// ---- scipy.optimize.linear_sum_assignment -------------------------------------------------

int augmenting_path(int nc, const std::vector<double>& cost, std::vector<double>& u,
                    std::vector<double>& v, std::vector<int>& path, std::vector<int>& row4col,
                    std::vector<double>& spc, int i, std::vector<char>& SR,
                    std::vector<char>& SC, std::vector<int>& remaining, double& min_val) {
  const double inf = std::numeric_limits<double>::infinity();
  min_val = 0;
  int num_remaining = nc;
  for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
  std::fill(SR.begin(), SR.end(), 0);
  std::fill(SC.begin(), SC.end(), 0);
  std::fill(spc.begin(), spc.end(), inf);
  int sink = -1;
  while (sink == -1) {
    int index = -1;
    double lowest = inf;
    SR[i] = 1;
    for (int it = 0; it < num_remaining; ++it) {
      int j = remaining[it];
      double r = min_val + cost[static_cast<size_t>(i) * nc + j] - u[i] - v[j];
      if (r < spc[j]) path[j] = i, spc[j] = r;
      if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) lowest = spc[j], index = it;
    }
    min_val = lowest;
    if (min_val == inf) return -1;
    int j = remaining[index];
    if (row4col[j] == -1) sink = j;
    else i = row4col[j];
    SC[j] = 1;
    remaining[index] = remaining[--num_remaining];
  }
  return sink;
}

}  // namespace

std::vector<std::pair<int, int>> linear_sum_assignment(const std::vector<double>& input,
                                                       int nr, int nc) {
  if (nr == 0 || nc == 0) return {};
  // Rows must not outnumber columns: solve the transpose otherwise, as scipy does.
  const bool transpose = nc < nr;
  std::vector<double> cost = input;
  if (transpose) {
    for (int i = 0; i < nr; ++i)
      for (int j = 0; j < nc; ++j) cost[static_cast<size_t>(j) * nr + i] = input[i * nc + j];
    std::swap(nr, nc);
  }
  std::vector<double> u(nr, 0), v(nc, 0), spc(nc);
  std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
  std::vector<char> SR(nr), SC(nc);
  for (int cur = 0; cur < nr; ++cur) {
    double min_val;
    int sink = augmenting_path(nc, cost, u, v, path, row4col, spc, cur, SR, SC, remaining,
                               min_val);
    if (sink < 0) throw std::runtime_error("cost matrix is infeasible");
    u[cur] += min_val;
    for (int i = 0; i < nr; ++i)
      if (SR[i] && i != cur) u[i] += min_val - spc[col4row[i]];
    for (int j = 0; j < nc; ++j)
      if (SC[j]) v[j] -= min_val - spc[j];
    int j = sink;
    while (true) {
      int i = path[j];
      row4col[j] = i;
      std::swap(col4row[i], j);
      if (i == cur) break;
    }
  }
  std::vector<std::pair<int, int>> out;
  for (int i = 0; i < nr; ++i)
    out.emplace_back(transpose ? col4row[i] : i, transpose ? i : col4row[i]);
  std::sort(out.begin(), out.end());
  return out;
}

// ---- ByteTrack (aerotrack/track/bytetrack.py) --------------------------------------------

void ByteTrack::Track::xyxy(double out[4]) const {
  double w = mean[2] * mean[3], h = mean[3];
  out[0] = mean[0] - w / 2, out[1] = mean[1] - h / 2;
  out[2] = mean[0] + w / 2, out[3] = mean[1] + h / 2;
}

ByteTrack::ByteTrack(double high, double low, double new_thresh, int buffer, double match,
                     bool fuse_score)
    : high_(high), low_(low), new_(new_thresh), match_(match), buffer_(buffer),
      fuse_(fuse_score) {}

ByteTrack::TrackPtr ByteTrack::make(const double* det) const {
  auto t = std::make_shared<Track>();
  double z[4];
  xyxy_to_xyah(det, z);
  kf_initiate(z, t->mean, t->cov);
  t->score = det[4], t->cls = static_cast<int>(det[5]);
  t->start_frame = t->frame_id = frame_id_;
  return t;
}

void ByteTrack::correct(Track& t, const double* det) const {
  double z[4];
  xyxy_to_xyah(det, z);
  kf_update(t.mean, t.cov, z);
  t.score = det[4], t.cls = static_cast<int>(det[5]), t.frame_id = frame_id_;
  t.state = kTracked, t.activated = true;
}

void ByteTrack::predict_pool(const Tracks& pool) const {
  for (const auto& t : pool) {
    if (t->state != kTracked) t->mean[7] = 0;  // lost: freeze the height
    kf_predict(t->mean, t->cov);
  }
}

std::vector<double> ByteTrack::cost(const Tracks& tracks, const std::vector<const double*>& dets,
                                    bool fuse) const {
  std::vector<double> c(tracks.size() * dets.size());
  for (size_t i = 0; i < tracks.size(); ++i) {
    double box[4];
    tracks[i]->xyxy(box);
    for (size_t j = 0; j < dets.size(); ++j) {
      double v = iou(box, dets[j]);
      c[i * dets.size() + j] = 1 - (fuse ? v * dets[j][4] : v);
    }
  }
  return c;
}

ByteTrack::Assignment ByteTrack::assign(const std::vector<double>& cost, int rows, int cols,
                                        double thresh) {
  Assignment a;
  std::vector<char> mr(rows, 0), mc(cols, 0);
  for (auto [r, c] : linear_sum_assignment(cost, rows, cols))
    if (cost[static_cast<size_t>(r) * cols + c] <= thresh) {
      a.pairs.emplace_back(r, c);
      mr[r] = mc[c] = 1;
    }
  for (int r = 0; r < rows; ++r)
    if (!mr[r]) a.rows.push_back(r);
  for (int c = 0; c < cols; ++c)
    if (!mc[c]) a.cols.push_back(c);
  return a;
}

std::vector<double> ByteTrack::update(const double* dets, int n) {
  ++frame_id_;
  std::vector<const double*> high, low;
  for (int k = 0; k < n; ++k) {
    const double* d = dets + 6 * k;
    if (!(d[2] > d[0] && d[3] > d[1])) continue;
    if (d[4] >= high_) high.push_back(d);
    else if (d[4] > low_) low.push_back(d);
  }

  Tracks unconfirmed, pool;
  for (const auto& t : tracked_) (t->activated ? pool : unconfirmed).push_back(t);
  pool.insert(pool.end(), lost_.begin(), lost_.end());
  predict_pool(pool);

  Tracks updated, refound, lost, removed;
  auto apply = [&](const TrackPtr& t, const double* det) {
    (t->state == kTracked ? updated : refound).push_back(t);
    correct(*t, det);
  };

  int nh = static_cast<int>(high.size());
  Assignment a = assign(cost(pool, high, fuse_), static_cast<int>(pool.size()), nh, match_);
  for (auto [i, j] : a.pairs) apply(pool[i], high[j]);

  Tracks remaining;
  for (int i : a.rows)
    if (pool[i]->state == kTracked) remaining.push_back(pool[i]);
  Assignment b = assign(cost(remaining, low, false), static_cast<int>(remaining.size()),
                        static_cast<int>(low.size()), 0.5);
  for (auto [i, j] : b.pairs) apply(remaining[i], low[j]);
  for (int i : b.rows) {
    remaining[i]->state = kLost;
    lost.push_back(remaining[i]);
  }

  std::vector<const double*> left;
  for (int j : a.cols) left.push_back(high[j]);
  Assignment c = assign(cost(unconfirmed, left, fuse_), static_cast<int>(unconfirmed.size()),
                        static_cast<int>(left.size()), 0.7);
  for (auto [i, j] : c.pairs) {
    correct(*unconfirmed[i], left[j]);
    updated.push_back(unconfirmed[i]);
  }
  for (int i : c.rows) {
    unconfirmed[i]->state = kRemoved;
    removed.push_back(unconfirmed[i]);
  }

  for (int j : c.cols)
    if (left[j][4] >= new_) {
      TrackPtr t = make(left[j]);
      t->track_id = next_id_++;
      t->activated = frame_id_ == 1;  // otherwise confirmed by a second match
      updated.push_back(t);
    }

  for (const auto& t : lost_)
    if (frame_id_ - t->frame_id > buffer_) {
      t->state = kRemoved;
      removed.push_back(t);
    }

  merge(updated, refound, lost, removed);
  return output();
}

std::vector<double> ByteTrack::predict_only() {
  ++frame_id_;
  Tracks pool;
  for (const auto& t : tracked_)
    if (t->activated) pool.push_back(t);
  pool.insert(pool.end(), lost_.begin(), lost_.end());
  predict_pool(pool);
  return output();
}

void ByteTrack::merge(const Tracks& updated, const Tracks& refound, const Tracks& lost,
                      const Tracks& removed) {
  auto ids = [](const Tracks& ts) {
    std::set<int> s;
    for (const auto& t : ts) s.insert(t->track_id);
    return s;
  };
  Tracks tracked;
  for (const auto& t : tracked_)
    if (t->state == kTracked) tracked.push_back(t);
  for (const Tracks* extra : {&updated, &refound}) {
    std::set<int> have = ids(tracked);
    for (const auto& t : *extra)
      if (!have.count(t->track_id)) tracked.push_back(t);
  }
  std::set<int> gone = ids(tracked), rem = ids(removed);
  gone.insert(rem.begin(), rem.end());
  Tracks pool_lost;
  for (const auto& t : lost_)
    if (!gone.count(t->track_id)) pool_lost.push_back(t);
  for (const auto& t : lost)
    if (!rem.count(t->track_id)) pool_lost.push_back(t);

  // A tracked and a lost track on the same object (IoU > 0.85): keep the older.
  std::set<size_t> drop_a, drop_b;
  for (size_t p = 0; p < tracked.size(); ++p) {
    double bp[4];
    tracked[p]->xyxy(bp);
    for (size_t q = 0; q < pool_lost.size(); ++q) {
      double bq[4];
      pool_lost[q]->xyxy(bq);
      if (1 - iou(bp, bq) < 0.15) {
        int age_p = tracked[p]->frame_id - tracked[p]->start_frame;
        int age_q = pool_lost[q]->frame_id - pool_lost[q]->start_frame;
        (age_p > age_q ? drop_b.insert(q) : drop_a.insert(p));
      }
    }
  }
  tracked_.clear();
  lost_.clear();
  for (size_t i = 0; i < tracked.size(); ++i)
    if (!drop_a.count(i)) tracked_.push_back(tracked[i]);
  for (size_t i = 0; i < pool_lost.size(); ++i)
    if (!drop_b.count(i)) lost_.push_back(pool_lost[i]);
}

std::vector<double> ByteTrack::output() const {
  std::vector<double> out;
  for (const auto& t : tracked_)
    if (t->activated) {
      double b[4];
      t->xyxy(b);
      out.insert(out.end(), {b[0], b[1], b[2], b[3], static_cast<double>(t->track_id),
                             t->score, static_cast<double>(t->cls)});
    }
  return out;
}

}  // namespace aerotrack

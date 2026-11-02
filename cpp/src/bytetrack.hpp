// ByteTrack and its Kalman filter in C++: a port of aerotrack.track.bytetrack and
// aerotrack.track.kalman, step for step (same thresholds, same pool order, same assignment
// algorithm as scipy's linear_sum_assignment), so the two give the same tracks on the same
// detections (tests/test_cpp.py). No CUDA: builds and is tested on any machine.
#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace aerotrack {

// Minimum-cost assignment of a rows x cols cost matrix (row-major): scipy's
// linear_sum_assignment (shortest augmenting path, Crouse 2016). Pairs by ascending row.
std::vector<std::pair<int, int>> linear_sum_assignment(const std::vector<double>& cost,
                                                       int rows, int cols);

class ByteTrack {
 public:
  ByteTrack(double high, double low, double new_thresh, int buffer, double match,
            bool fuse_score);

  // dets: n rows of x1, y1, x2, y2, score, class. Returns rows of
  // x1, y1, x2, y2, track id, score, class for the active tracks.
  std::vector<double> update(const double* dets, int n);
  std::vector<double> predict_only();

 private:
  struct Track {
    double mean[8], cov[64];
    double score;
    int cls, start_frame, frame_id, track_id = 0, state = 1;
    bool activated = false;
    void xyxy(double out[4]) const;
  };
  using TrackPtr = std::shared_ptr<Track>;
  using Tracks = std::vector<TrackPtr>;
  struct Assignment {
    std::vector<std::pair<int, int>> pairs;
    std::vector<int> rows, cols;  // unmatched
  };

  TrackPtr make(const double* det) const;
  void correct(Track& t, const double* det) const;
  void predict_pool(const Tracks& pool) const;
  std::vector<double> cost(const Tracks& tracks, const std::vector<const double*>& dets,
                           bool fuse) const;
  static Assignment assign(const std::vector<double>& cost, int rows, int cols, double thresh);
  void merge(const Tracks& updated, const Tracks& refound, const Tracks& lost,
             const Tracks& removed);
  std::vector<double> output() const;

  double high_, low_, new_, match_;
  int buffer_;
  bool fuse_;
  Tracks tracked_, lost_;
  int frame_id_ = 0, next_id_ = 1;
};

}  // namespace aerotrack

// Python module `aerotrack_cpp`: the C++ runtime behind Detector(backend="cpp").
// The host stages are exposed on their own so tests can compare them with the Python ones
// on any machine; TrtDetector only exists in a build with TensorRT (Linux + NVIDIA).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "bytetrack.hpp"
#include "pipeline.hpp"
#ifdef AEROTRACK_WITH_TRT
#include <NvInferVersion.h>

#include "trt_detector.hpp"
#endif

namespace py = pybind11;
using Image = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using Raw = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

void check_bgr(const Image& img) {
  if (img.ndim() != 3 || img.shape(2) != 3) throw std::invalid_argument("expected H x W x 3");
}

py::array_t<float> detections(const std::vector<float>& rows) {
  py::array_t<float> out({static_cast<py::ssize_t>(rows.size() / 6), py::ssize_t{6}});
  std::copy(rows.begin(), rows.end(), out.mutable_data());
  return out;
}

py::array_t<double> tracks(const std::vector<double>& rows) {
  py::array_t<double> out({static_cast<py::ssize_t>(rows.size() / 7), py::ssize_t{7}});
  std::copy(rows.begin(), rows.end(), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(aerotrack_cpp, m) {
  m.def("letterbox", [](const Image& img, int size) {
    check_bgr(img);
    py::array_t<uint8_t> out({py::ssize_t{size}, py::ssize_t{size}, py::ssize_t{3}});
    aerotrack::letterbox(img.data(), static_cast<int>(img.shape(0)),
                         static_cast<int>(img.shape(1)), size, out.mutable_data());
    return out;
  }, py::arg("img"), py::arg("size"));

  m.def("postprocess", [](const Raw& raw, int size, int h0, int w0, float conf, double iou,
                          int max_det) {
    if (raw.ndim() != 2) throw std::invalid_argument("expected (4 + nc) x anchors");
    return detections(aerotrack::postprocess(raw.data(), static_cast<int>(raw.shape(0)),
                                             static_cast<int>(raw.shape(1)), size, h0, w0,
                                             conf, iou, max_det));
  }, py::arg("raw"), py::arg("size"), py::arg("h0"), py::arg("w0"), py::arg("conf"),
     py::arg("iou"), py::arg("max_det"));

  m.def("linear_sum_assignment", [](const Doubles& cost) {
    if (cost.ndim() != 2) throw std::invalid_argument("expected a 2-D cost matrix");
    int r = static_cast<int>(cost.shape(0)), c = static_cast<int>(cost.shape(1));
    std::vector<double> v(cost.data(), cost.data() + cost.size());
    std::vector<int> rows, cols;
    for (auto [i, j] : aerotrack::linear_sum_assignment(v, r, c)) rows.push_back(i), cols.push_back(j);
    return py::make_tuple(py::array_t<int>(rows.size(), rows.data()),
                          py::array_t<int>(cols.size(), cols.data()));
  }, py::arg("cost"));

  // Same constructor and methods as aerotrack.track.bytetrack.ByteTrack.
  py::class_<aerotrack::ByteTrack>(m, "ByteTrack")
      .def(py::init<double, double, double, int, double, bool>(), py::arg("high") = 0.25,
           py::arg("low") = 0.1, py::arg("new") = 0.25, py::arg("buffer") = 30,
           py::arg("match") = 0.8, py::arg("fuse_score") = true)
      .def("update", [](aerotrack::ByteTrack& t, const Doubles& dets) {
        if (dets.size() && (dets.ndim() != 2 || dets.shape(1) != 6))
          throw std::invalid_argument("expected N x 6 detections");
        return tracks(t.update(dets.data(), static_cast<int>(dets.size() / 6)));
      }, py::arg("dets"))
      .def("predict_only", [](aerotrack::ByteTrack& t) { return tracks(t.predict_only()); });

  m.attr("compiler") = AEROTRACK_COMPILER;
#ifdef AEROTRACK_WITH_TRT
  m.attr("with_tensorrt") = true;
  m.attr("tensorrt_headers") = std::to_string(NV_TENSORRT_MAJOR) + "." +
                               std::to_string(NV_TENSORRT_MINOR) + "." +
                               std::to_string(NV_TENSORRT_PATCH) + "." +
                               std::to_string(NV_TENSORRT_BUILD);
  m.def("tensorrt_library", [] {  // the libnvinfer actually loaded: MMmmpp, e.g. 101601
    return getInferLibVersion();
  });

  py::class_<aerotrack::TrtDetector>(m, "TrtDetector")
      .def(py::init<const std::string&, int, float, double, int, int, bool>(),
           py::arg("engine"), py::arg("imgsz"), py::arg("conf"), py::arg("iou"),
           py::arg("max_det"), py::arg("device") = 0, py::arg("gpu_preprocess") = false)
      .def("preprocess", [](aerotrack::TrtDetector& d, const Image& img) {
        check_bgr(img);
        d.preprocess(img.data(), static_cast<int>(img.shape(0)), static_cast<int>(img.shape(1)));
      })
      .def("infer", &aerotrack::TrtDetector::infer)
      .def("postprocess", [](aerotrack::TrtDetector& d) { return detections(d.postprocess()); })
      .def_property_readonly("channels", &aerotrack::TrtDetector::channels)
      .def_property_readonly("anchors", &aerotrack::TrtDetector::anchors)
      .def_property_readonly("input_dtype", &aerotrack::TrtDetector::input_dtype)
      .def_property_readonly("output_dtype", &aerotrack::TrtDetector::output_dtype);
#else
  m.attr("with_tensorrt") = false;
#endif
}

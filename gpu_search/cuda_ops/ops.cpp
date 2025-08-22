// ops.cpp - LibTorch operator registration
#include <torch/torch.h>
#include <torch/script.h>

// Forward declarations from CUDA files
at::Tensor i8dot512_scores_cuda(const at::Tensor& q, const at::Tensor& db);
at::Tensor i8dot512_batch_cuda(const at::Tensor& queries, const at::Tensor& db);
at::Tensor build_pq_lut_cuda(const at::Tensor& q_rot, const at::Tensor& cb);
at::Tensor adc_scan_cuda(const at::Tensor& codes, const at::Tensor& lut);
std::tuple<at::Tensor, at::Tensor> gather_ivf_codes_cuda(
    const at::Tensor& codes,
    const at::Tensor& ids,
    const at::Tensor& list_ids,
    const at::Tensor& list_offsets);

// Register custom operators
TORCH_LIBRARY(gobed_ann, m) {
  m.def("i8dot512_scores(Tensor q, Tensor db) -> Tensor");
  m.def("i8dot512_batch(Tensor queries, Tensor db) -> Tensor");
  m.def("build_pq_lut(Tensor q_rot, Tensor cb) -> Tensor");
  m.def("adc_scan(Tensor codes, Tensor lut) -> Tensor");
  m.def("gather_ivf_codes(Tensor codes, Tensor ids, Tensor list_ids, Tensor list_offsets) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gobed_ann, CUDA, m) {
  m.impl("i8dot512_scores", &i8dot512_scores_cuda);
  m.impl("i8dot512_batch", &i8dot512_batch_cuda);
  m.impl("build_pq_lut", &build_pq_lut_cuda);
  m.impl("adc_scan", &adc_scan_cuda);
  m.impl("gather_ivf_codes", &gather_ivf_codes_cuda);
}

// Python bindings for testing
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("i8dot512_scores", &i8dot512_scores_cuda, "INT8 dot product scores");
  m.def("i8dot512_batch", &i8dot512_batch_cuda, "Batch INT8 dot products");
  m.def("build_pq_lut", &build_pq_lut_cuda, "Build PQ lookup table");
  m.def("adc_scan", &adc_scan_cuda, "ADC scan with LUT");
  m.def("gather_ivf_codes", &gather_ivf_codes_cuda, "Gather codes from IVF lists");
}
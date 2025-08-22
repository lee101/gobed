// ops_cuda.cpp - CUDA operator registration that properly links implementations
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
at::Tensor check_cuda_capabilities();
at::Tensor test_version_compatibility(const at::Tensor& db, const at::Tensor& query);

// Direct registration without separate library/impl split
static auto registry = torch::RegisterOperators()
    .op("gobed_ann::i8dot512_scores(Tensor q, Tensor db) -> Tensor",
        &i8dot512_scores_cuda)
    .op("gobed_ann::i8dot512_batch(Tensor queries, Tensor db) -> Tensor",
        &i8dot512_batch_cuda)
    .op("gobed_ann::build_pq_lut(Tensor q_rot, Tensor cb) -> Tensor",
        &build_pq_lut_cuda)
    .op("gobed_ann::adc_scan(Tensor codes, Tensor lut) -> Tensor",
        &adc_scan_cuda)
    .op("gobed_ann::gather_ivf_codes(Tensor codes, Tensor ids, Tensor list_ids, Tensor list_offsets) -> (Tensor, Tensor)",
        &gather_ivf_codes_cuda)
    .op("gobed_ann::check_cuda_capabilities() -> Tensor",
        &check_cuda_capabilities)
    .op("gobed_ann::test_version_compatibility(Tensor db, Tensor query) -> Tensor",
        &test_version_compatibility);
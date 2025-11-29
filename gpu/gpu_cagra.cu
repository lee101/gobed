// Minimal GPU IVF-gated candidate scoring for CAGRA-style pipeline scaffolding
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <cstring>

extern "C" {

typedef struct gpu_cagra_t {
    int nlist;
    int degree;
    int dim;
    int total;

    // list layout: vectors are stored grouped by list in ascending list id
    // d_offsets has length nlist+1; list l spans [offsets[l], offsets[l+1]) in the ordered arrays
    int* d_offsets;

    // ordered arrays (by list)
    signed char* d_vecs;
    float* d_scales;
    int* d_ids;
} gpu_cagra_t;

gpu_cagra_t* gpu_cagra_create(int nlist, int degree, int vector_dim) {
    gpu_cagra_t* h = (gpu_cagra_t*)calloc(1, sizeof(gpu_cagra_t));
    if (!h) return nullptr;
    h->nlist = nlist; h->degree = degree; h->dim = vector_dim; h->total = 0;
    cudaMalloc(&h->d_offsets, sizeof(int) * (nlist + 1));
    cudaMemset(h->d_offsets, 0, sizeof(int) * (nlist + 1));
    return h;
}

void gpu_cagra_destroy(gpu_cagra_t* h) {
    if (!h) return;
    if (h->d_offsets) cudaFree(h->d_offsets);
    if (h->d_vecs) cudaFree(h->d_vecs);
    if (h->d_scales) cudaFree(h->d_scales);
    if (h->d_ids) cudaFree(h->d_ids);
    free(h);
}

// host helper to build offsets from list_ids and reorder arrays
static void build_layout(const int* list_ids, int nlist, int n, int* h_offsets, int* h_order) {
    // count per list
    std::vector<int> counts(nlist, 0);
    for (int i = 0; i < n; i++) counts[list_ids[i]]++;
    h_offsets[0] = 0;
    for (int l = 0; l < nlist; l++) h_offsets[l+1] = h_offsets[l] + counts[l];
    // current write cursor per list
    std::vector<int> cur(h_offsets, h_offsets + nlist);
    for (int i = 0; i < n; i++) {
        int lid = list_ids[i];
        int pos = cur[lid]++;
        h_order[pos] = i; // i maps to new position pos
    }
}

int gpu_cagra_build(gpu_cagra_t* h,
                    const signed char* vectors, const float* scales, const int* ids,
                    const int* list_ids, int num_vectors) {
    if (!h || num_vectors <= 0) return 0;
    // compute layout on host
    std::vector<int> h_offsets(h->nlist + 1, 0);
    std::vector<int> h_order(num_vectors, 0);
    std::vector<int> h_list_ids(num_vectors, 0);
    for (int i = 0; i < num_vectors; i++) h_list_ids[i] = list_ids[i];
    build_layout(h_list_ids.data(), h->nlist, num_vectors, h_offsets.data(), h_order.data());

    // allocate device arrays
    size_t vbytes = (size_t)num_vectors * (size_t)h->dim * sizeof(signed char);
    size_t sbytes = (size_t)num_vectors * sizeof(float);
    size_t ibytes = (size_t)num_vectors * sizeof(int);
    cudaMalloc(&h->d_vecs, vbytes);
    cudaMalloc(&h->d_scales, sbytes);
    cudaMalloc(&h->d_ids, ibytes);

    // reorder on host for simplicity, then copy; keeps build simple and deterministic
    std::vector<signed char> h_vecs(vbytes);
    std::vector<float> h_scales(num_vectors);
    std::vector<int> h_ids(num_vectors);
    for (int newpos = 0; newpos < num_vectors; newpos++) {
        int old = h_order[newpos];
        memcpy(&h_vecs[(size_t)newpos * h->dim], &vectors[(size_t)old * h->dim], (size_t)h->dim);
        h_scales[newpos] = scales[old];
        h_ids[newpos] = ids[old];
    }
    cudaMemcpy(h->d_vecs, h_vecs.data(), vbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_scales, h_scales.data(), sbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_ids, h_ids.data(), ibytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_offsets, h_offsets.data(), sizeof(int)*(h->nlist+1), cudaMemcpyHostToDevice);
    h->total = num_vectors;
    return 1;
}

// simple int8 dot product kernel: scores[i] = dot(q, vec[i]) * qscale * scale[i]
__global__ void int8_dot_kernel(const signed char* __restrict__ q,
                                const signed char* __restrict__ vecs,
                                const float* __restrict__ scales,
                                float* __restrict__ out,
                                int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const signed char* v = vecs + (size_t)i * dim;
    int sum = 0;
    #pragma unroll 4
    for (int d = 0; d < dim; d++) { sum += int(q[d]) * int(v[d]); }
    out[i] = (float)sum * scales[i];
}

// Search across vectors in provided lists; k top results returned.
int gpu_cagra_search_lists(gpu_cagra_t* h,
                           const signed char* query, float qscale,
                           const int* list_ids, int nprobe, int k,
                           int* result_ids, float* result_scores) {
    if (!h || h->total == 0 || nprobe <= 0 || k <= 0) return 0;
    // gather span across all probe lists
    std::vector<int> h_offsets(h->nlist + 1);
    cudaMemcpy(h_offsets.data(), h->d_offsets, sizeof(int)*(h->nlist+1), cudaMemcpyDeviceToHost);
    int total_cand = 0;
    // treat list_ids as host pointer (Go passes host memory); compute total candidates
    for (int i = 0; i < nprobe; i++) {
        int lid = ((const int*)list_ids)[i];
        if (lid < 0 || lid >= h->nlist) continue;
        total_cand += (h_offsets[lid+1] - h_offsets[lid]);
    }
    if (total_cand == 0) return 0;
    // compute scores for concatenated candidate spans by running kernel per list
    thrust::device_vector<float> scores(total_cand);
    thrust::device_vector<int>    idx(total_cand);
    int cursor = 0;
    for (int i = 0; i < nprobe; i++) {
        int lid = ((const int*)list_ids)[i];
        if (lid < 0 || lid >= h->nlist) continue;
        int beg = h_offsets[lid], end = h_offsets[lid+1];
        int cnt = end - beg;
        if (cnt <= 0) continue;
        // launch kernel for this slice; write into scores[cursor:cursor+cnt]
        float* d_scores_slice = thrust::raw_pointer_cast(scores.data()) + cursor;
        int threads = 256; int blocks = (cnt + threads - 1)/threads;
        int8_dot_kernel<<<blocks, threads>>>(query, h->d_vecs + (size_t)beg * h->dim, h->d_scales + beg, d_scores_slice, h->dim, cnt);
        // fill indices with absolute positions in ordered arrays
        thrust::sequence(thrust::device, idx.begin() + cursor, idx.begin() + cursor + cnt, beg);
        cursor += cnt;
    }
    // scale scores by qscale
    if (qscale != 1.0f) {
        thrust::transform(scores.begin(), scores.end(), scores.begin(), [=] __device__(float x){ return x * qscale; });
    }
    // top-k (descending)
    // Use sort_by_key on negative scores to get descending quickly
    thrust::device_vector<float> neg_scores = scores;
    thrust::transform(neg_scores.begin(), neg_scores.end(), neg_scores.begin(), [=] __device__(float x){ return -x; });
    thrust::sort_by_key(neg_scores.begin(), neg_scores.end(), idx.begin());
    int take = k; if (take > total_cand) take = total_cand;
    // gather only top-k ids on device, then copy to host
    thrust::device_vector<int> top_idx(take);
    thrust::copy(idx.begin(), idx.begin()+take, top_idx.begin());
    thrust::device_ptr<const int> d_ids_ptr(h->d_ids);
    thrust::device_vector<int> top_ids(take);
    thrust::gather(top_idx.begin(), top_idx.end(), d_ids_ptr, top_ids.begin());
    std::vector<int> h_ids(take);
    std::vector<float> h_scores(take);
    thrust::copy(top_ids.begin(), top_ids.end(), h_ids.begin());
    thrust::copy(scores.begin(), scores.begin()+take, h_scores.begin());
    for (int i = 0; i < take; i++) {
        result_ids[i] = h_ids[i];
        result_scores[i] = h_scores[i];
    }
    return take;
}

} // extern "C"

/*
 * CAGRA wrapper header for gobed
 */

#ifndef CAGRA_WRAPPER_H
#define CAGRA_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct cagra_wrapper_t cagra_wrapper_t;

// Performance stats
typedef struct {
    double build_time_ms;
    double search_time_ms;
    double serialize_time_ms;
    double deserialize_time_ms;
    size_t index_memory_bytes;
    int num_vectors;
    int dim;
} cagra_stats_t;

// Core API
cagra_wrapper_t* cagra_create(int max_vectors, int dim);
void cagra_destroy(cagra_wrapper_t* wrapper);

// Index operations
int cagra_build_index(
    cagra_wrapper_t* wrapper,
    const int8_t* vectors,
    const float* scales,
    int num_vectors
);

// Search operations
int cagra_search(
    cagra_wrapper_t* wrapper,
    const int8_t* query,
    float query_scale,
    int k,
    uint32_t* neighbors,
    float* distances
);

int cagra_search_batch(
    cagra_wrapper_t* wrapper,
    const int8_t* queries,
    const float* query_scales,
    int num_queries,
    int k,
    uint32_t* neighbors,
    float* distances
);

// Persistence
int cagra_serialize_index(cagra_wrapper_t* wrapper, const char* filepath);
int cagra_deserialize_index(cagra_wrapper_t* wrapper, const char* filepath);

// Stats
void cagra_get_stats(cagra_wrapper_t* wrapper, cagra_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif // CAGRA_WRAPPER_H
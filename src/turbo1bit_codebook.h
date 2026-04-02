/**
 * turbo1bit_codebook.h — Pre-computed Lloyd-Max codebooks for TurboQuant KV cache compression.
 *
 * These codebooks are the optimal scalar quantizers for the distribution of coordinates
 * of unit-norm vectors after random orthogonal rotation. At high dimension d, each
 * coordinate follows approximately N(0, 1/d), but the exact distribution is:
 *   f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
 *
 * Codebooks are pre-computed by the TurboQuant Python library via Lloyd-Max iteration.
 * We embed them as static arrays to avoid runtime computation.
 */

#ifndef TURBO1BIT_CODEBOOK_H
#define TURBO1BIT_CODEBOOK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum supported configurations
#define T1B_MAX_BITS 4
#define T1B_MAX_CLUSTERS (1 << T1B_MAX_BITS) // 16

struct t1b_codebook {
    int      d;                              // embedding dimension
    int      bits;                           // bits per coordinate
    int      n_clusters;                     // 2^bits
    float    centroids[T1B_MAX_CLUSTERS];    // sorted centroid values
    float    boundaries[T1B_MAX_CLUSTERS+1]; // decision boundaries (n_clusters+1)
    float    mse_per_coord;                  // achieved MSE per coordinate
};

// Get pre-computed codebook for given (d, bits).
// Returns NULL if no codebook exists for this configuration.
const struct t1b_codebook * t1b_get_codebook(int d, int bits);

#ifdef __cplusplus
}
#endif

#endif // TURBO1BIT_CODEBOOK_H

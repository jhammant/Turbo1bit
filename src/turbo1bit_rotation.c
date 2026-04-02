/**
 * turbo1bit_rotation.c — Random orthogonal rotation via QR decomposition.
 *
 * We use a simple xoshiro256** PRNG seeded deterministically per layer,
 * then generate a Gaussian matrix via Box-Muller, and compute QR via
 * modified Gram-Schmidt (sufficient for our purposes — the matrix quality
 * only needs to decorrelate coordinates, not achieve cryptographic randomness).
 *
 * For production use with Accelerate framework on macOS, we use LAPACK's
 * dgeqrf/dorgqr for numerically stable QR. But we provide a self-contained
 * fallback via modified Gram-Schmidt.
 */

#include "turbo1bit_rotation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define T1B_HAS_LAPACK 1
#else
#define T1B_HAS_LAPACK 0
#endif

// ── PRNG (xoshiro256**) ─────────────────────────────────────────────

typedef struct {
    uint64_t s[4];
} t1b_rng;

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(t1b_rng *r) {
    uint64_t result = rotl(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl(r->s[3], 45);
    return result;
}

static void rng_seed(t1b_rng *r, uint64_t seed) {
    // SplitMix64 to initialize state from single seed
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        r->s[i] = z;
    }
}

static double rng_uniform(t1b_rng *r) {
    return (double)(rng_next(r) >> 11) * 0x1.0p-53;
}

static double rng_normal(t1b_rng *r) {
    // Box-Muller transform
    double u1 = rng_uniform(r);
    double u2 = rng_uniform(r);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ── Rotation structure ──────────────────────────────────────────────

struct t1b_rotation {
    int   dim;
    float *Pi;  // rotation matrix, row-major, dim x dim
    float *S;   // QJL projection matrix, row-major, dim x dim
};

// ── QR via modified Gram-Schmidt ────────────────────────────────────

// Always compiled — used as fallback even when LAPACK is available
static void gram_schmidt_qr(float *Q, int n) {
    // In-place: Q starts as the random matrix, ends as orthogonal Q
    // Column-major for easier column operations
    float *col_i = (float *)malloc(n * sizeof(float));
    if (!col_i) return;

    for (int i = 0; i < n; i++) {
        // Copy column i
        for (int r = 0; r < n; r++) {
            col_i[r] = Q[r * n + i]; // row-major: Q[r][i]
        }

        // Subtract projections onto previous columns
        for (int j = 0; j < i; j++) {
            float dot = 0.0f;
            for (int r = 0; r < n; r++) {
                dot += col_i[r] * Q[r * n + j];
            }
            for (int r = 0; r < n; r++) {
                col_i[r] -= dot * Q[r * n + j];
            }
        }

        // Normalize
        float norm = 0.0f;
        for (int r = 0; r < n; r++) {
            norm += col_i[r] * col_i[r];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int r = 0; r < n; r++) {
                col_i[r] /= norm;
            }
        }

        // Write back
        for (int r = 0; r < n; r++) {
            Q[r * n + i] = col_i[r];
        }
    }
    free(col_i);
}

// Generate orthogonal matrix via QR decomposition of random Gaussian matrix
static void generate_orthogonal(float *Q, int dim, uint64_t seed) {
    t1b_rng rng;
    rng_seed(&rng, seed);

    // Fill with Gaussian random values
    for (int i = 0; i < dim * dim; i++) {
        Q[i] = (float)rng_normal(&rng);
    }

#if T1B_HAS_LAPACK
    // Use LAPACK QR factorization for numerical stability
    // LAPACK works in column-major, so we transpose, factor, transpose back
    // Actually, since random Gaussian is symmetric in distribution, we can just
    // use row-major directly and the result is still a valid random orthogonal matrix.

    int n = dim;
    int lda = dim;
    int lwork = dim * 4;
    float *tau = (float *)malloc(dim * sizeof(float));
    float *work = (float *)malloc(lwork * sizeof(float));
    int info = 0;

    if (!tau || !work) {
        free(tau);
        free(work);
        // Fallback to Gram-Schmidt
        gram_schmidt_qr(Q, dim);
        return;
    }

    // Transpose to column-major for LAPACK
    float *col_major = (float *)malloc(dim * dim * sizeof(float));
    if (!col_major) {
        free(tau); free(work);
        gram_schmidt_qr(Q, dim);
        return;
    }
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            col_major[j * dim + i] = Q[i * dim + j];

    // QR factorization
    sgeqrf_(&n, &n, col_major, &lda, tau, work, &lwork, &info);
    if (info != 0) {
        free(tau); free(work); free(col_major);
        gram_schmidt_qr(Q, dim);
        return;
    }

    // Fix signs to match Python: Q = Q * sign(diag(R))
    // After sgeqrf, R is in the upper triangle of col_major
    float *diag_signs = (float *)malloc(dim * sizeof(float));
    if (diag_signs) {
        for (int i = 0; i < dim; i++) {
            float r_ii = col_major[i * dim + i];
            diag_signs[i] = (r_ii >= 0.0f) ? 1.0f : -1.0f;
        }
    }

    // Generate Q matrix
    sorgqr_(&n, &n, &n, col_major, &lda, tau, work, &lwork, &info);

    // Apply sign correction
    if (diag_signs) {
        for (int j = 0; j < dim; j++) {
            for (int i = 0; i < dim; i++) {
                col_major[j * dim + i] *= diag_signs[j];
            }
        }
    }

    // Transpose back to row-major
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            Q[i * dim + j] = col_major[j * dim + i];

    free(diag_signs);
    free(tau);
    free(work);
    free(col_major);
#else
    gram_schmidt_qr(Q, dim);
#endif
}

// Generate Gaussian random matrix (not orthogonalized)
static void generate_gaussian(float *S, int dim, uint64_t seed) {
    t1b_rng rng;
    rng_seed(&rng, seed);
    for (int i = 0; i < dim * dim; i++) {
        S[i] = (float)rng_normal(&rng);
    }
}

// ── Public API ──────────────────────────────────────────────────────

t1b_rotation * t1b_rotation_create(int dim, int layer_idx) {
    t1b_rotation *rot = (t1b_rotation *)calloc(1, sizeof(t1b_rotation));
    if (!rot) return NULL;

    rot->dim = dim;
    rot->Pi = (float *)malloc(dim * dim * sizeof(float));
    rot->S  = (float *)malloc(dim * dim * sizeof(float));

    if (!rot->Pi || !rot->S) {
        t1b_rotation_free(rot);
        return NULL;
    }

    uint64_t rotation_seed = 42 + (uint64_t)layer_idx * 7;
    uint64_t qjl_seed      = rotation_seed + 1000;

    generate_orthogonal(rot->Pi, dim, rotation_seed);
    generate_gaussian(rot->S, dim, qjl_seed);

    return rot;
}

void t1b_rotation_free(t1b_rotation *rot) {
    if (!rot) return;
    free(rot->Pi);
    free(rot->S);
    free(rot);
}

int t1b_rotation_dim(const t1b_rotation *rot) {
    return rot->dim;
}

void t1b_rotate_forward(const t1b_rotation *rot, const float *x, float *y) {
    int d = rot->dim;
    // y = x @ Pi^T, i.e. y[i] = sum_j x[j] * Pi[i][j]
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        const float *row = rot->Pi + i * d;
        for (int j = 0; j < d; j++) {
            sum += x[j] * row[j];
        }
        y[i] = sum;
    }
}

void t1b_rotate_backward(const t1b_rotation *rot, const float *y, float *x) {
    int d = rot->dim;
    // x = y @ Pi, i.e. x[i] = sum_j y[j] * Pi[j][i]
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += y[j] * rot->Pi[j * d + i];
        }
        x[i] = sum;
    }
}

void t1b_qjl_project(const t1b_rotation *rot, const float *x, float *out) {
    int d = rot->dim;
    // out = x @ S^T, i.e. out[i] = sum_j x[j] * S[i][j]
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        const float *row = rot->S + i * d;
        for (int j = 0; j < d; j++) {
            sum += x[j] * row[j];
        }
        out[i] = sum;
    }
}

const float * t1b_get_qjl_matrix(const t1b_rotation *rot) {
    return rot->S;
}

const float * t1b_get_rotation_matrix(const t1b_rotation *rot) {
    return rot->Pi;
}

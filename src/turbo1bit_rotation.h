/**
 * turbo1bit_rotation.h — Random orthogonal rotation and QJL projection matrices.
 *
 * The rotation matrix Pi is generated via QR decomposition of a random Gaussian matrix.
 * The QJL matrix S has i.i.d. N(0,1) entries (not orthogonal).
 * Both are deterministic from their seed, generated once per layer.
 *
 * For head_dim=128: each matrix is 128x128 = 64KB in float32 — negligible.
 */

#ifndef TURBO1BIT_ROTATION_H
#define TURBO1BIT_ROTATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for rotation state (holds Pi and S matrices for one layer)
typedef struct t1b_rotation t1b_rotation;

// Create rotation state for a given dimension and layer.
// Seeds: rotation = 42 + layer_idx * 7, QJL = rotation_seed + 1000
// Returns NULL on allocation failure.
t1b_rotation * t1b_rotation_create(int dim, int layer_idx);

// Free rotation state.
void t1b_rotation_free(t1b_rotation *rot);

// Get dimension
int t1b_rotation_dim(const t1b_rotation *rot);

// Apply forward rotation: y[i] = sum_j x[j] * Pi[i][j]  (y = x @ Pi^T)
// x and y must be float arrays of length dim.
void t1b_rotate_forward(const t1b_rotation *rot, const float *x, float *y);

// Apply inverse rotation: x[i] = sum_j y[j] * Pi[j][i]  (x = y @ Pi)
void t1b_rotate_backward(const t1b_rotation *rot, const float *y, float *x);

// Project through QJL matrix: out[i] = sum_j x[j] * S[i][j]  (out = x @ S^T)
void t1b_qjl_project(const t1b_rotation *rot, const float *x, float *out);

// Get pointer to the QJL matrix S (row-major, dim x dim). Used for sketching queries.
const float * t1b_get_qjl_matrix(const t1b_rotation *rot);

// Get pointer to the rotation matrix Pi (row-major, dim x dim).
const float * t1b_get_rotation_matrix(const t1b_rotation *rot);

#ifdef __cplusplus
}
#endif

#endif // TURBO1BIT_ROTATION_H

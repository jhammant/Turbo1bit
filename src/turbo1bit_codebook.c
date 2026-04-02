/**
 * turbo1bit_codebook.c — Static Lloyd-Max codebooks ported from TurboQuant JSON files.
 *
 * These values are pre-computed by turboquant/codebook.py and embedded directly.
 * Each codebook is optimal for the Beta distribution on [-1,1] arising from
 * random rotation of d-dimensional unit vectors.
 */

#include "turbo1bit_codebook.h"
#include <stddef.h>

// ── d=128, bits=1: 2 centroids ──────────────────────────────────────
static const struct t1b_codebook cb_d128_b1 = {
    .d = 128, .bits = 1, .n_clusters = 2,
    .centroids   = { -0.08862269254527579f, 0.08862269254527579f },
    .boundaries  = { -1.0f, 0.0f, 1.0f },
    .mse_per_coord = 0.005600f,
};

// ── d=128, bits=2: 4 centroids ──────────────────────────────────────
static const struct t1b_codebook cb_d128_b2 = {
    .d = 128, .bits = 2, .n_clusters = 4,
    .centroids   = { -0.1330401982533685f, -0.039990945215356365f,
                      0.039990945215356365f, 0.1330401982533685f },
    .boundaries  = { -1.0f, -0.08651557173436243f, 0.0f,
                      0.08651557173436243f, 1.0f },
    .mse_per_coord = 0.0009062505493759921f,
};

// ── d=128, bits=3: 8 centroids ──────────────────────────────────────
static const struct t1b_codebook cb_d128_b3 = {
    .d = 128, .bits = 3, .n_clusters = 8,
    .centroids   = { -0.188390613802078f, -0.11813298369899362f,
                     -0.06658059531595685f, -0.021602468667239208f,
                      0.021602468667239208f, 0.06658059531595685f,
                      0.11813298369899362f, 0.188390613802078f },
    .boundaries  = { -1.0f, -0.1532617987505358f, -0.09235678950747524f,
                     -0.04409153199159803f, 0.0f, 0.04409153199159803f,
                      0.09235678950747524f, 0.1532617987505358f, 1.0f },
    .mse_per_coord = 0.00026535879813162263f,
};

// ── d=64, bits=1: 2 centroids ───────────────────────────────────────
static const struct t1b_codebook cb_d64_b1 = {
    .d = 64, .bits = 1, .n_clusters = 2,
    .centroids   = { -0.12533141373155001f, 0.12533141373155001f },
    .boundaries  = { -1.0f, 0.0f, 1.0f },
    .mse_per_coord = 0.011200f,
};

// ── d=64, bits=2: 4 centroids ───────────────────────────────────────
static const struct t1b_codebook cb_d64_b2 = {
    .d = 64, .bits = 2, .n_clusters = 4,
    .centroids   = { -0.18816041454849518f, -0.056533413731550014f,
                      0.056533413731550014f, 0.18816041454849518f },
    .boundaries  = { -1.0f, -0.12234691413502260f, 0.0f,
                      0.12234691413502260f, 1.0f },
    .mse_per_coord = 0.001812501098751984f,
};

// ── d=64, bits=3: 8 centroids ───────────────────────────────────────
static const struct t1b_codebook cb_d64_b3 = {
    .d = 64, .bits = 3, .n_clusters = 8,
    .centroids   = { -0.26639624951624563f, -0.16704087498592150f,
                     -0.09413735994750316f, -0.030546484263698640f,
                      0.030546484263698640f, 0.09413735994750316f,
                      0.16704087498592150f, 0.26639624951624563f },
    .boundaries  = { -1.0f, -0.21671856225108356f, -0.13058911746671233f,
                     -0.06234192210560090f, 0.0f, 0.06234192210560090f,
                      0.13058911746671233f, 0.21671856225108356f, 1.0f },
    .mse_per_coord = 0.00053071759626324526f,
};

// ── d=128, bits=4: 16 centroids ─────────────────────────────────────
static const struct t1b_codebook cb_d128_b4 = {
    .d = 128, .bits = 4, .n_clusters = 16,
    .centroids   = { -0.23762718673095357f, -0.18079372947217434f,
                     -0.14176165429673310f, -0.11024706538276363f,
                     -0.08279256667309579f, -0.05774453560525709f,
                     -0.03413402823112088f, -0.01129649814274393f,
                      0.01129649814274384f,  0.03413402823112079f,
                      0.05774453560525705f,  0.08279256667309574f,
                      0.11024706538276359f,  0.14176165429673304f,
                      0.18079372947217426f,  0.23762718673095345f },
    .boundaries  = { -1.0f, -0.20921045810156397f, -0.16127769188445373f,
                     -0.12600435983974836f, -0.09651981602792971f,
                     -0.07026855113917643f, -0.04593928191818898f,
                     -0.02271526318693240f,  0.0f,
                      0.02271526318693231f,  0.04593928191818892f,
                      0.07026855113917640f,  0.09651981602792967f,
                      0.12600435983974830f,  0.16127769188445365f,
                      0.20921045810156386f,  1.0f },
    .mse_per_coord = 7.27718115381205e-05f,
};

// ── Lookup table ────────────────────────────────────────────────────
struct cb_entry {
    int d;
    int bits;
    const struct t1b_codebook *cb;
};

static const struct cb_entry cb_table[] = {
    {  64, 1, &cb_d64_b1  },
    {  64, 2, &cb_d64_b2  },
    {  64, 3, &cb_d64_b3  },
    { 128, 1, &cb_d128_b1 },
    { 128, 2, &cb_d128_b2 },
    { 128, 3, &cb_d128_b3 },
    { 128, 4, &cb_d128_b4 },
};

static const int cb_table_len = sizeof(cb_table) / sizeof(cb_table[0]);

const struct t1b_codebook * t1b_get_codebook(int d, int bits) {
    for (int i = 0; i < cb_table_len; i++) {
        if (cb_table[i].d == d && cb_table[i].bits == bits) {
            return cb_table[i].cb;
        }
    }
    return NULL;
}

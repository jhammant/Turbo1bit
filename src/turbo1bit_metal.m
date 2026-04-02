/**
 * turbo1bit_metal.m — Metal host-side dispatch for Turbo1Bit GPU kernels.
 *
 * Objective-C file for Metal API interaction.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "turbo1bit_metal.h"
#include <string.h>

struct t1b_metal_ctx {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;
    id<MTLComputePipelineState> ps_mse_score;
    id<MTLComputePipelineState> ps_qjl_score;
    id<MTLComputePipelineState> ps_fused_attn;
    id<MTLComputePipelineState> ps_dequant_values;
    id<MTLComputePipelineState> ps_matvec;
};

bool t1b_metal_available(void) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil;
}

static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                                  id<MTLLibrary> library,
                                                  NSString *name) {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"[Turbo1Bit] Metal function '%@' not found", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&error];
    if (error) {
        NSLog(@"[Turbo1Bit] Failed to create pipeline '%@': %@", name, error);
    }
    return ps;
}

t1b_metal_ctx * t1b_metal_init(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return NULL;

    t1b_metal_ctx *ctx = (t1b_metal_ctx *)calloc(1, sizeof(t1b_metal_ctx));
    if (!ctx) return NULL;

    ctx->device = device;
    ctx->queue = [device newCommandQueue];

    // Load shader library from .metallib or compile from source
    NSError *error = nil;

    // Search for Metal shader source and compile at runtime
    // This avoids needing the Metal toolchain installed at build time
    NSArray *searchPaths = @[
        @"src/turbo1bit_metal.metal",
        @"../src/turbo1bit_metal.metal",
        @"../../src/turbo1bit_metal.metal",
        @"../../../src/turbo1bit_metal.metal",
    ];

    // Also try relative to executable
    NSString *execDir = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
    NSArray *execRelPaths = @[
        [execDir stringByAppendingPathComponent:@"../../src/turbo1bit_metal.metal"],
        [execDir stringByAppendingPathComponent:@"../../../src/turbo1bit_metal.metal"],
    ];
    NSMutableArray *allPaths = [NSMutableArray arrayWithArray:searchPaths];
    [allPaths addObjectsFromArray:execRelPaths];

    for (NSString *path in allPaths) {
        NSString *source = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:nil];
        if (source) {
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = YES;
            ctx->library = [device newLibraryWithSource:source options:opts error:&error];
            if (ctx->library) {
                NSLog(@"[Turbo1Bit] Loaded Metal shaders from: %@", path);
                break;
            }
        }
    }

    if (!ctx->library) {
        NSLog(@"[Turbo1Bit] Failed to load Metal shaders: %@", error);
        free(ctx);
        return NULL;
    }

    // Create pipeline states
    ctx->ps_mse_score     = make_pipeline(device, ctx->library, @"t1b_mse_score");
    ctx->ps_qjl_score     = make_pipeline(device, ctx->library, @"t1b_qjl_score");
    ctx->ps_fused_attn    = make_pipeline(device, ctx->library, @"t1b_fused_attn");
    ctx->ps_dequant_values= make_pipeline(device, ctx->library, @"t1b_dequant_values");
    ctx->ps_matvec        = make_pipeline(device, ctx->library, @"t1b_matvec");

    return ctx;
}

void t1b_metal_free(t1b_metal_ctx *ctx) {
    if (!ctx) return;
    // ARC handles Metal object release
    free(ctx);
}

// ── Helper: create buffer from host data ────────────────────────────

static id<MTLBuffer> make_buf(id<MTLDevice> dev, const void *data, size_t size) {
    if (size == 0) return [dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    return [dev newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_buf_rw(id<MTLDevice> dev, size_t size) {
    return [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
}

// ── GPU Operations ──────────────────────────────────────────────────

void t1b_metal_mse_score(
    t1b_metal_ctx *ctx,
    const float *query_rot,
    const uint8_t *mse_packed,
    const float *norms,
    float *scores_out,
    uint32_t n_tokens,
    uint32_t head_dim,
    uint32_t packed_dim,
    uint32_t bits)
{
    if (!ctx || !ctx->ps_mse_score || n_tokens == 0) return;

    id<MTLBuffer> buf_query  = make_buf(ctx->device, query_rot, head_dim * sizeof(float));
    id<MTLBuffer> buf_packed = make_buf(ctx->device, mse_packed, (size_t)n_tokens * packed_dim);
    id<MTLBuffer> buf_norms  = make_buf(ctx->device, norms, n_tokens * sizeof(float));
    id<MTLBuffer> buf_scores = make_buf_rw(ctx->device, n_tokens * sizeof(float));

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx->ps_mse_score];
    [enc setBuffer:buf_query  offset:0 atIndex:0];
    [enc setBuffer:buf_packed offset:0 atIndex:1];
    [enc setBuffer:buf_norms  offset:0 atIndex:2];
    [enc setBuffer:buf_scores offset:0 atIndex:3];
    [enc setBytes:&n_tokens   length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&head_dim   length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&packed_dim length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&bits       length:sizeof(uint32_t) atIndex:7];

    MTLSize grid = MTLSizeMake(n_tokens, 1, 1);
    MTLSize group = MTLSizeMake(MIN(256, n_tokens), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(scores_out, buf_scores.contents, n_tokens * sizeof(float));
}

void t1b_metal_qjl_score(
    t1b_metal_ctx *ctx,
    const float *q_sketch,
    const uint8_t *qjl_packed,
    const float *residual_norms,
    float *scores_inout,
    uint32_t n_tokens,
    uint32_t head_dim,
    uint32_t sign_packed_dim)
{
    if (!ctx || !ctx->ps_qjl_score || n_tokens == 0) return;

    id<MTLBuffer> buf_sketch = make_buf(ctx->device, q_sketch, head_dim * sizeof(float));
    id<MTLBuffer> buf_packed = make_buf(ctx->device, qjl_packed, (size_t)n_tokens * sign_packed_dim);
    id<MTLBuffer> buf_rnorms = make_buf(ctx->device, residual_norms, n_tokens * sizeof(float));
    // scores_inout already has MSE scores — load them
    id<MTLBuffer> buf_scores = make_buf(ctx->device, scores_inout, n_tokens * sizeof(float));

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx->ps_qjl_score];
    [enc setBuffer:buf_sketch offset:0 atIndex:0];
    [enc setBuffer:buf_packed offset:0 atIndex:1];
    [enc setBuffer:buf_rnorms offset:0 atIndex:2];
    [enc setBuffer:buf_scores offset:0 atIndex:3];
    [enc setBytes:&n_tokens   length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&head_dim   length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&sign_packed_dim length:sizeof(uint32_t) atIndex:6];

    MTLSize grid = MTLSizeMake(n_tokens, 1, 1);
    MTLSize group = MTLSizeMake(MIN(256, n_tokens), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(scores_inout, buf_scores.contents, n_tokens * sizeof(float));
}

void t1b_metal_matvec(
    t1b_metal_ctx *ctx,
    const float *x,
    const float *M,
    float *y,
    uint32_t dim)
{
    if (!ctx || !ctx->ps_matvec) return;

    id<MTLBuffer> buf_x = make_buf(ctx->device, x, dim * sizeof(float));
    id<MTLBuffer> buf_M = make_buf(ctx->device, M, (size_t)dim * dim * sizeof(float));
    id<MTLBuffer> buf_y = make_buf_rw(ctx->device, dim * sizeof(float));

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx->ps_matvec];
    [enc setBuffer:buf_x offset:0 atIndex:0];
    [enc setBuffer:buf_M offset:0 atIndex:1];
    [enc setBuffer:buf_y offset:0 atIndex:2];
    [enc setBytes:&dim   length:sizeof(uint32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(dim, 1, 1);
    MTLSize group = MTLSizeMake(MIN(256, dim), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(y, buf_y.contents, dim * sizeof(float));
}

void t1b_metal_fused_attn(
    t1b_metal_ctx *ctx,
    const float *query_rot,
    const float *q_sketch,
    const uint8_t *mse_packed,
    const uint8_t *qjl_packed,
    const float *key_norms,
    const float *residual_norms,
    const uint8_t *val_packed,
    const float *val_scales,
    const float *val_zeros,
    const float *buf_keys,
    const float *buf_values,
    float *output,
    uint32_t n_compressed,
    uint32_t n_buffered,
    uint32_t head_dim,
    float attn_scale)
{
    if (!ctx || !ctx->ps_fused_attn) return;

    uint32_t mse_packed_dim = head_dim / 4;  // 2-bit = 4 per byte
    uint32_t sign_packed_dim = (head_dim + 7) / 8;
    uint32_t val_packed_dim = head_dim / 4;  // 2-bit values
    uint32_t n_groups = head_dim / 32;
    uint32_t group_size = 32;

    id<MTLBuffer> buf_qrot   = make_buf(ctx->device, query_rot, head_dim * sizeof(float));
    id<MTLBuffer> buf_qsk    = make_buf(ctx->device, q_sketch, head_dim * sizeof(float));
    id<MTLBuffer> buf_msep   = make_buf(ctx->device, mse_packed, (size_t)n_compressed * mse_packed_dim);
    id<MTLBuffer> buf_qjlp   = make_buf(ctx->device, qjl_packed, (size_t)n_compressed * sign_packed_dim);
    id<MTLBuffer> buf_knorms = make_buf(ctx->device, key_norms, n_compressed * sizeof(float));
    id<MTLBuffer> buf_rnorms = make_buf(ctx->device, residual_norms, n_compressed * sizeof(float));
    id<MTLBuffer> buf_valp   = make_buf(ctx->device, val_packed, (size_t)n_compressed * val_packed_dim);
    id<MTLBuffer> buf_vscale = make_buf(ctx->device, val_scales, (size_t)n_compressed * n_groups * sizeof(float));
    id<MTLBuffer> buf_vzeros = make_buf(ctx->device, val_zeros, (size_t)n_compressed * n_groups * sizeof(float));
    id<MTLBuffer> buf_out    = make_buf_rw(ctx->device, head_dim * sizeof(float));
    id<MTLBuffer> buf_bk     = make_buf(ctx->device, buf_keys, (size_t)n_buffered * head_dim * sizeof(float));
    id<MTLBuffer> buf_bv     = make_buf(ctx->device, buf_values, (size_t)n_buffered * head_dim * sizeof(float));

    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ctx->ps_fused_attn];
    [enc setBuffer:buf_qrot   offset:0 atIndex:0];
    [enc setBuffer:buf_qsk    offset:0 atIndex:1];
    [enc setBuffer:buf_msep   offset:0 atIndex:2];
    [enc setBuffer:buf_qjlp   offset:0 atIndex:3];
    [enc setBuffer:buf_knorms offset:0 atIndex:4];
    [enc setBuffer:buf_rnorms offset:0 atIndex:5];
    [enc setBuffer:buf_valp   offset:0 atIndex:6];
    [enc setBuffer:buf_vscale offset:0 atIndex:7];
    [enc setBuffer:buf_vzeros offset:0 atIndex:8];
    [enc setBuffer:buf_out    offset:0 atIndex:9];
    [enc setBytes:&n_compressed   length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&head_dim       length:sizeof(uint32_t) atIndex:11];
    [enc setBytes:&mse_packed_dim length:sizeof(uint32_t) atIndex:12];
    [enc setBytes:&sign_packed_dim length:sizeof(uint32_t) atIndex:13];
    [enc setBytes:&val_packed_dim length:sizeof(uint32_t) atIndex:14];
    [enc setBytes:&n_groups       length:sizeof(uint32_t) atIndex:15];
    [enc setBytes:&group_size     length:sizeof(uint32_t) atIndex:16];
    [enc setBytes:&attn_scale     length:sizeof(float)    atIndex:17];
    [enc setBuffer:buf_bk     offset:0 atIndex:18];
    [enc setBuffer:buf_bv     offset:0 atIndex:19];
    [enc setBytes:&n_buffered     length:sizeof(uint32_t) atIndex:20];

    // Single threadgroup — fused kernel processes sequentially for now
    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize group = MTLSizeMake(1, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(output, buf_out.contents, head_dim * sizeof(float));
}

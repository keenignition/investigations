// AUTO-GENERATED — do not edit.
// Source: configs/archs/sm_75.yml
// Re-generate: python scripts/gen_kernel_config.py

#ifndef SOFTMAX_KERNEL_CONFIG_H
#define SOFTMAX_KERNEL_CONFIG_H

// ---------------------------------------------------------------------------
// fused_warp kernel
// ---------------------------------------------------------------------------
#define FUSED_WARP_THREADBLOCK_SIZE 128
#define FUSED_WARP_MIN_N 1024
#define FUSED_WARP_MAX_N 4096

#define FUSED_WARP_SIZES(F)  \
  F(1024) \
  F(2048) \
  F(4096)

// ---------------------------------------------------------------------------
// fused_block kernel
// ---------------------------------------------------------------------------
#define FUSED_BLOCK_SIZE 1024
#define FUSED_BLOCK_MIN_N 4096
#define FUSED_BLOCK_MAX_N 65536

#define FUSED_BLOCK_SIZES(F)  \
  F(4096) \
  F(8192) \
  F(16384) \
  F(32768) \
  F(65536)

// ---------------------------------------------------------------------------
// online kernel
// ---------------------------------------------------------------------------
#define ONLINE_MIN_N 1024
#define ONLINE_MAX_N 262144
#define ONLINE_2PASS_MIN_NP 32

#define ONLINE_SIZES(F)  \
  F(1024, 256) \
  F(2048, 256) \
  F(4096, 1024) \
  F(8192, 1024) \
  F(16384, 1024) \
  F(32768, 1024) \
  F(65536, 1024) \
  F(131072, 1024) \
  F(262144, 1024)

// ---------------------------------------------------------------------------
// online_v2 kernel
// ---------------------------------------------------------------------------
#define ONLINE_V2_MIN_N 1024
#define ONLINE_V2_MAX_N 262144
#define ONLINE_V2_SINGLE_MAX_N 65536

#define ONLINE_V2_SINGLE_SIZES(F)  \
  F(1024, 256) \
  F(2048, 256) \
  F(4096, 1024) \
  F(8192, 1024) \
  F(16384, 1024) \
  F(32768, 1024) \
  F(65536, 1024)

#define V2_TARGET_NP 4
#define V2_MULTI_BS 1024
#define V2_SPLIT_THRESHOLD 16

#endif /* SOFTMAX_KERNEL_CONFIG_H */

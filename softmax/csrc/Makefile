CSRC_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
OUT_DIR   := $(CSRC_DIR)out
BIN_DIR   := $(CSRC_DIR)bin

CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d ' ' | sed 's/\.//' | sed 's/^/sm_/')

ifeq ($(CUDA_ARCH),sm_)
  $(error Could not detect GPU arch. Set CUDA_ARCH=sm_XX)
endif

MAXRREGCOUNT ?= 64

NVCC      := nvcc
NVCCFLAGS := -O3                    \
             -std=c++17             \
             -use_fast_math         \
             -lineinfo              \
             -Xptxas=-v             \
             -arch=$(CUDA_ARCH)     \
             -maxrregcount=$(MAXRREGCOUNT) \
             -DSOFTMAX_STANDALONE   \
             -I$(CSRC_DIR)

LDFLAGS   := -lcurand

# ---------------------------------------------------------------------------
# Profiler flags
# ---------------------------------------------------------------------------
NCU_SKIP  ?= 5
NCU_COUNT ?= 1

NCU_FLAGS  := --set full -f \
              --launch-skip $(NCU_SKIP) --launch-count $(NCU_COUNT)

NSYS_FLAGS := -t cuda,nvtx -f true

ACTION := $(firstword $(MAKECMDGOALS))
KERNEL := $(word 2,$(MAKECMDGOALS))

ifneq ($(KERNEL),)
$(KERNEL): ;
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: compile ncu nsys gpu clean help

gpu: $(BIN_DIR)/gpu
	$(BIN_DIR)/gpu

$(BIN_DIR)/gpu: $(CSRC_DIR)gpu.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -I$(CSRC_DIR) $< -o $@

compile: $(BIN_DIR)/$(KERNEL)

$(BIN_DIR)/%: $(CSRC_DIR)%.cu $(CSRC_DIR)utils.h $(CSRC_DIR)kernel_config.h
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[build] $(notdir $@)  arch=$(CUDA_ARCH)  maxrregcount=$(MAXRREGCOUNT)"

ncu: $(BIN_DIR)/$(KERNEL)
	@mkdir -p $(OUT_DIR)
	ncu $(NCU_FLAGS) -o $(OUT_DIR)/$(KERNEL) $(BIN_DIR)/$(KERNEL)
	@echo "[ncu] report → $(OUT_DIR)/$(KERNEL).ncu-rep"

nsys: $(BIN_DIR)/$(KERNEL)
	@mkdir -p $(OUT_DIR)
	nsys profile $(NSYS_FLAGS) -o $(OUT_DIR)/$(KERNEL) $(BIN_DIR)/$(KERNEL)
	@echo "[nsys] report → $(OUT_DIR)/$(KERNEL).nsys-rep"

clean:
	rm -rf $(BIN_DIR) $(OUT_DIR)

help:
	@echo "make gpu                build and print GPU device properties"
	@echo "make compile <kernel>   build standalone binary"
	@echo "make ncu     <kernel>   profile with Nsight Compute  → out/<kernel>.ncu-rep"
	@echo "make nsys    <kernel>   profile with Nsight Systems  → out/<kernel>.nsys-rep"
	@echo "make clean              remove bin/ and out/"
	@echo ""
	@echo "kernels: naive  warpReduction  fused  fusedBlock  online  online_v2"
	@echo ""
	@echo "overrides:"
	@echo "  CUDA_ARCH=sm_89       (detected: $(CUDA_ARCH))"
	@echo "  MAXRREGCOUNT=128      (current:  $(MAXRREGCOUNT))"
	@echo "  NCU_SKIP=5            warmup launches to skip"
	@echo "  NCU_COUNT=1           launches to capture"

SRC += src/opt/fxchcuda/FxchCudaSCHashTable.c

# CUDA source files
CUDA_SRC += src/opt/fxchcuda/Kernel.cu

# CUDA compilation flags
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -fPIC

# CUDA object files
CUDA_OBJS = $(CUDA_SRC:.cu=.o)

# Rule for compiling CUDA files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

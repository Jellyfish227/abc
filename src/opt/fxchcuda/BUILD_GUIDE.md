# FXCH CUDA Build Guide

## Prerequisites

1. **NVIDIA GPU** with compute capability 6.0 or higher
2. **CUDA Toolkit** (version 11.0 or later recommended)
   - Download from: https://developer.nvidia.com/cuda-downloads
3. **C/C++ Compiler**
   - GCC 8.0 or later (Linux)
   - Clang (macOS with CUDA support)
   - MSVC 2019 or later (Windows)

## Installation Steps

### 1. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

### 2. Compilation

#### Option A: Using Make (Recommended)

Add the following to your main Makefile:

```makefile
# Include CUDA module
include src/opt/fxchcuda/module.make

# Add CUDA libraries to linker
LIBS += -lcudart -L/usr/local/cuda/lib64

# Compile CUDA files
$(CUDA_OBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) -I. -Isrc $< -c -o $@

# Add CUDA objects to main build
abc: $(OBJS) $(CUDA_OBJS)
	$(CXX) -o $@ $^ $(LIBS)
```

#### Option B: Manual Compilation

```bash
# Compile CUDA kernel
nvcc -O3 -arch=sm_60 -Xcompiler -fPIC \
     -I. -Isrc \
     -c src/opt/fxchcuda/Kernel.cu \
     -o src/opt/fxchcuda/Kernel.o

# Compile C wrapper (as C++)
g++ -O3 -fPIC \
    -I. -Isrc \
    -c src/opt/fxchcuda/FxchCudaSCHashTable.c \
    -o src/opt/fxchcuda/FxchCudaSCHashTable.o

# Link with your main application
g++ -o abc [...other objects...] \
    src/opt/fxchcuda/Kernel.o \
    src/opt/fxchcuda/FxchCudaSCHashTable.o \
    -lcudart -L/usr/local/cuda/lib64
```

### 3. Runtime Configuration

#### Set CUDA Library Path

```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
```

## Troubleshooting

### Issue: "nvcc: command not found"

**Solution**: Add CUDA to your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### Issue: "cannot find -lcudart"

**Solution**: Ensure CUDA library path is set:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Issue: "undefined reference to `cudaMalloc`"

**Solution**: Ensure you're linking against the CUDA runtime:
```bash
# Add to linker flags
-lcudart -L/usr/local/cuda/lib64
```

### Issue: "CUDA error: out of memory"

**Symptoms**: 
```
CUDA error in Kernel.cu:XXX: out of memory
CUDA kernel failed, falling back to CPU
```

**Solutions**:
1. **Reduce cube data size**: Process smaller benchmarks
2. **Use CPU fallback**: Set `usingGpu = 0`
3. **Upgrade GPU**: Requires more VRAM

### Issue: Slow performance despite GPU

**Possible causes**:
1. **Small bins**: GPU overhead dominates for bins < 10 entries
2. **Data transfer**: Cache not working properly
3. **Old GPU**: Compute capability < 6.0

**Solutions**:
1. Profile with `nvprof` to identify bottlenecks
2. Check cache hit rate (add debug prints)
3. Use CPU for small operations

## Architecture-Specific Compilation

### Finding Your GPU Architecture

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Common architectures:
- **sm_60**: Pascal (GTX 1000 series, Tesla P100)
- **sm_70**: Volta (Tesla V100)
- **sm_75**: Turing (RTX 2000 series)
- **sm_80**: Ampere (RTX 3000 series, A100)
- **sm_86**: Ampere (RTX 3050/3060)
- **sm_89**: Ada Lovelace (RTX 4000 series)
- **sm_90**: Hopper (H100)

### Optimizing for Your GPU

Update `module.make`:
```makefile
# For RTX 3080 (sm_86)
NVCCFLAGS = -O3 -arch=sm_86 -Xcompiler -fPIC

# For multiple architectures
NVCCFLAGS = -O3 -gencode arch=compute_60,code=sm_60 \
                -gencode arch=compute_75,code=sm_75 \
                -gencode arch=compute_86,code=sm_86 \
                -Xcompiler -fPIC
```

## Performance Tuning

### 1. Adjust Thread Block Size

Edit `Kernel.cu`:
```cpp
// Line ~389
int threadsPerBlock = 256;  // Try 128, 256, 512
```

**Guidelines**:
- Small bins: 128 threads
- Medium bins: 256 threads (default)
- Large bins: 512 threads

### 2. Enable Fast Math

Add to `NVCCFLAGS`:
```makefile
NVCCFLAGS += --use_fast_math
```

### 3. Profile with NSight

```bash
# Profile GPU activity
nsys profile --stats=true ./abc -c "fxchcuda benchmark.aig"

# Detailed kernel analysis
ncu --set full ./abc -c "fxchcuda benchmark.aig"
```

## Platform-Specific Notes

### Linux
- Works out of the box with standard CUDA installation
- Ensure `gcc` version is compatible with CUDA version
- Check compatibility: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

### macOS
- CUDA support deprecated after macOS 10.13
- Last supported CUDA version: 10.2
- Consider using Metal or OpenCL instead
- Or use Linux VM/Docker with GPU passthrough

### Windows
- Install Visual Studio 2019 or later
- Install CUDA Toolkit with VS integration
- Use MSBuild or CMake for compilation

Example CMakeLists.txt:
```cmake
find_package(CUDA REQUIRED)

cuda_add_library(fxchcuda
    src/opt/fxchcuda/Kernel.cu
    src/opt/fxchcuda/FxchCudaSCHashTable.c
)

target_link_libraries(abc fxchcuda ${CUDA_LIBRARIES})
```

## Verification

### Test Installation

```bash
# Run with GPU disabled (baseline)
./abc -c "read benchmark.aig; fxch; print_stats"

# Run with GPU enabled
./abc -c "read benchmark.aig; fxchcuda; print_stats"
```

### Verify Correctness

Results should be identical between CPU and GPU versions. If not:
1. Check for CUDA errors in output
2. Enable debug prints in `Kernel.cu`
3. Run with smaller test cases
4. File a bug report with reproduction steps

### Benchmark Performance

```bash
#!/bin/bash
echo "Testing FXCH CUDA Performance"

for bench in benchmarks/*.aig; do
    echo "Benchmark: $bench"
    
    # CPU version
    echo -n "CPU: "
    time ./abc -c "read $bench; fxch; quit" 2>&1 | grep "Time"
    
    # GPU version
    echo -n "GPU: "
    time ./abc -c "read $bench; fxchcuda; quit" 2>&1 | grep "Time"
    
    echo "---"
done
```

## Expected Performance

Based on profiling data (100k1.txt):

| Metric | CPU (Original) | GPU (Optimized) | Speedup |
|--------|---------------|----------------|---------|
| Entry comparisons | 406.6M calls | 406.6M calls | 1x |
| Sequential time | 718.30s | ~7-72s | 10-100x |
| Total time | 742s | ~30-150s | 5-25x |
| Memory usage | Low | Higher | - |

**Note**: Actual speedup depends on:
- GPU model and compute capability
- Bin size distribution
- Data transfer overhead
- Cache efficiency

## Debugging Tips

### Enable Verbose CUDA Output

Add to `Kernel.cu` after error checks:
```cpp
#define CUDA_DEBUG 1

#ifdef CUDA_DEBUG
    printf("[CUDA] Allocated %ld bytes for cube data\n", cubeDataSize);
    printf("[CUDA] Launching kernel with %d blocks, %d threads\n", 
           blocks, threadsPerBlock);
#endif
```

### Check Kernel Execution

```cpp
// After kernel launch
cudaDeviceSynchronize();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
}
```

### Memory Leak Detection

```bash
# Use cuda-memcheck
cuda-memcheck --leak-check full ./abc -c "fxchcuda test.aig"
```

## Further Optimization

See `README_CUDA.md` for:
- Data transfer caching strategies
- Batch processing techniques
- Asynchronous execution
- Shared memory optimization
- Multi-GPU support

## Support

For issues or questions:
1. Check NVIDIA CUDA forums: https://forums.developer.nvidia.com/c/accelerated-computing/cuda
2. Review ABC system documentation
3. Post detailed error messages and GPU info


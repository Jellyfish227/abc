#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void FxchCudaCompareKernel(
    const int* candidateCubeData,
    const int* candidateCubeOffsets,
    const int* candidateCubeLengths,
    const uint32_t* candidateIds,
    const int* candidateILit0,
    const int* candidateILit1,
    const int* candidateOutputData,
    int numCandidates,
    int nSizeOutputID,
    const int* refCubeData,
    int refCubeLength,
    uint32_t refId,
    int refILit0,
    int refILit1,
    const int* refOutputData,
    int* results )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= numCandidates )
        return;

    int len0 = candidateCubeLengths[idx];
    int len1 = refCubeLength;
    const int* cube0 = candidateCubeData ? candidateCubeData + candidateCubeOffsets[idx] : NULL;
    const int* cube1 = refCubeData;

    if ( len0 <= 0 || len1 <= 0 || cube0 == NULL || cube1 == NULL )
    {
        results[idx] = 0;
        return;
    }

    if ( cube0[0] != cube1[0] )
    {
        results[idx] = 0;
        return;
    }

    if ( candidateIds[idx] != refId )
    {
        results[idx] = 0;
        return;
    }

    if ( nSizeOutputID <= 0 || candidateOutputData == NULL || refOutputData == NULL )
    {
        results[idx] = 0;
        return;
    }

    const int* output0 = candidateOutputData + idx * nSizeOutputID;
    const int* output1 = refOutputData;
    int i;
    int hasIntersection = 0;
    for ( i = 0; i < nSizeOutputID; ++i )
    {
        if ( ( output0[i] & output1[i] ) != 0 )
        {
            hasIntersection = 1;
            break;
        }
    }

    if ( hasIntersection == 0 )
    {
        results[idx] = 0;
        return;
    }

    int lit0_a = candidateILit0[idx];
    int lit0_b = candidateILit1[idx];
    int lit1_a = refILit0;
    int lit1_b = refILit1;

    if ( lit0_b > 0 && lit1_b > 0 )
    {
        if ( lit0_a >= len0 || lit0_b >= len0 || lit1_a >= len1 || lit1_b >= len1 )
        {
            results[idx] = 0;
            return;
        }

        int val00 = cube0[lit0_a];
        int val01 = cube0[lit0_b];
        int val10 = cube1[lit1_a];
        int val11 = cube1[lit1_b];

        if ( val00 == val10 || val00 == val11 || val01 == val10 || val01 == val11 )
        {
            results[idx] = 0;
            return;
        }
    }

    int pos0 = 0;
    int pos1 = 0;

    while ( true )
    {
        while ( pos0 < len0 && ( ( lit0_a > 0 && pos0 == lit0_a ) || ( lit0_b > 0 && pos0 == lit0_b ) ) )
            pos0++;
        while ( pos1 < len1 && ( ( lit1_a > 0 && pos1 == lit1_a ) || ( lit1_b > 0 && pos1 == lit1_b ) ) )
            pos1++;

        if ( pos0 >= len0 || pos1 >= len1 )
            break;

        if ( cube0[pos0] != cube1[pos1] )
        {
            results[idx] = 0;
            return;
        }

        pos0++;
        pos1++;
    }

    while ( pos0 < len0 && ( ( lit0_a > 0 && pos0 == lit0_a ) || ( lit0_b > 0 && pos0 == lit0_b ) ) )
        pos0++;
    while ( pos1 < len1 && ( ( lit1_a > 0 && pos1 == lit1_a ) || ( lit1_b > 0 && pos1 == lit1_b ) ) )
        pos1++;

    if ( pos0 != len0 || pos1 != len1 )
    {
        results[idx] = 0;
        return;
    }

    results[idx] = 1;
}

extern "C" cudaError_t FxchCudaLaunchCompareKernel(
    const int* d_candidateCubeData,
    const int* d_candidateCubeOffsets,
    const int* d_candidateCubeLengths,
    const uint32_t* d_candidateIds,
    const int* d_candidateILit0,
    const int* d_candidateILit1,
    const int* d_candidateOutputData,
    int numCandidates,
    int nSizeOutputID,
    const int* d_refCubeData,
    int refCubeLength,
    uint32_t refId,
    int refILit0,
    int refILit1,
    const int* d_refOutputData,
    int* d_results )
{
    if ( numCandidates <= 0 )
        return cudaSuccess;

    dim3 block( 256 );
    dim3 grid( ( numCandidates + block.x - 1 ) / block.x );

    FxchCudaCompareKernel<<< grid, block >>>( d_candidateCubeData,
                                              d_candidateCubeOffsets,
                                              d_candidateCubeLengths,
                                              d_candidateIds,
                                              d_candidateILit0,
                                              d_candidateILit1,
                                              d_candidateOutputData,
                                              numCandidates,
                                              nSizeOutputID,
                                              d_refCubeData,
                                              refCubeLength,
                                              refId,
                                              refILit0,
                                              refILit1,
                                              d_refOutputData,
                                              d_results );

    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        return err;

    return cudaDeviceSynchronize();
}


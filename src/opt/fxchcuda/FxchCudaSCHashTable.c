/**CFile****************************************************************

  FileName    [ FxchCudaSCHashTable.c ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ Sub-cubes hash table implementation ]

  Author      [ Yu Ching Hei, Chan Eugene ]

  Affiliation [ CUHK ]

  Date        [ Ver. 1.0. Started - October 22, 2025. ]

  Revision    []

***********************************************************************/
#include "FxchCuda.h"

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>

static inline void MurmurHash3_x86_32 ( const void* key,
                                        int len,
                                        uint32_t seed,
                                        void* out )
{
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint8_t * tail;
    uint32_t k1;

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);
    int i;

    for(i = -nblocks; i; i++)
    {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> (32 - 15));
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> (32 - 13));
        h1 = h1*5+0xe6546b64;
    }

    tail = (const uint8_t*)(data + nblocks*4);

    k1 = 0;

    switch(len & 3)
    {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
              k1 *= c1; k1 = (k1 << 15) | (k1 >> (32 - 15)); k1 *= c2; h1 ^= k1;
    };

    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    *(uint32_t*)out = h1;
}

#if (__GNUC__ >= 8)
  #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

// -----------------------------------------------------------------------------
// Forward declarations for CUDA helpers implemented in Kernel.cu
// -----------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t FxchCudaLaunchCompareKernel(
    const int* d_candidate_cube_data,
    const int* d_candidate_cube_offsets,
    const int* d_candidate_cube_lengths,
    const uint32_t* d_candidate_ids,
    const int* d_candidate_iLit0,
    const int* d_candidate_iLit1,
    const int* d_candidate_output_data,
    int numCandidates,
    int nSizeOutputID,
    const int* d_ref_cube_data,
    int ref_cube_length,
    uint32_t ref_id,
    int ref_iLit0,
    int ref_iLit1,
    const int* d_ref_output_data,
    int* d_results);
#ifdef __cplusplus
}
#endif

// -----------------------------------------------------------------------------
// Local helpers
// -----------------------------------------------------------------------------

static inline int FxchCuda_SCHashTableEntryCompareCPU( Fxch_SCHashTable_t* pSCHashTable,
                                                       Vec_Wec_t* vCubes,
                                                       Fxch_SubCube_t* pSCData0,
                                                       Fxch_SubCube_t* pSCData1 )
{
    Vec_Int_t* vCube0 = Vec_WecEntry( vCubes, pSCData0->iCube );
    Vec_Int_t* vCube1 = Vec_WecEntry( vCubes, pSCData1->iCube );

    int* pOutputID0 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pSCData0->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
    int* pOutputID1 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pSCData1->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
    int i, Result = 0;

    if ( !Vec_IntSize( vCube0 ) ||
         !Vec_IntSize( vCube1 ) ||
         Vec_IntEntry( vCube0, 0 ) != Vec_IntEntry( vCube1, 0 ) ||
         pSCData0->Id != pSCData1->Id )
        return 0;

    for ( i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID && Result == 0; i++ )
        Result = ( pOutputID0[i] & pOutputID1[i] );

    if ( Result == 0 )
        return 0;

    Vec_IntClear( &pSCHashTable->vSubCube0 );
    Vec_IntClear( &pSCHashTable->vSubCube1 );

    if ( pSCData0->iLit1 > 0 && pSCData1->iLit1 > 0 &&
         ( Vec_IntEntry( vCube0, pSCData0->iLit0 ) == Vec_IntEntry( vCube1, pSCData1->iLit0 ) ||
           Vec_IntEntry( vCube0, pSCData0->iLit0 ) == Vec_IntEntry( vCube1, pSCData1->iLit1 ) ||
           Vec_IntEntry( vCube0, pSCData0->iLit1 ) == Vec_IntEntry( vCube1, pSCData1->iLit0 ) ||
           Vec_IntEntry( vCube0, pSCData0->iLit1 ) == Vec_IntEntry( vCube1, pSCData1->iLit1 ) ) )
        return 0;

    if ( pSCData0->iLit0 > 0 )
        Vec_IntAppendSkip( &pSCHashTable->vSubCube0, vCube0, pSCData0->iLit0 );
    else
        Vec_IntAppend( &pSCHashTable->vSubCube0, vCube0 );

    if ( pSCData1->iLit0 > 0 )
        Vec_IntAppendSkip( &pSCHashTable->vSubCube1, vCube1, pSCData1->iLit0 );
    else
        Vec_IntAppend( &pSCHashTable->vSubCube1, vCube1 );

    if ( pSCData0->iLit1 > 0 )
        Vec_IntDrop( &pSCHashTable->vSubCube0,
                     pSCData0->iLit0 < pSCData0->iLit1 ? pSCData0->iLit1 - 1 : pSCData0->iLit1 );

    if ( pSCData1->iLit1 > 0 )
        Vec_IntDrop( &pSCHashTable->vSubCube1,
                     pSCData1->iLit0 < pSCData1->iLit1 ? pSCData1->iLit1 - 1 : pSCData1->iLit1 );

    return Vec_IntEqual( &pSCHashTable->vSubCube0, &pSCHashTable->vSubCube1 );
}

static int FxchCuda_SCHashTableCompareBatchGpu( Fxch_SCHashTable_t* pSCHashTable,
                                                Vec_Wec_t* vCubes,
                                                Fxch_SubCube_t* pReference,
                                                Fxch_SubCube_t* pCandidates,
                                                int nCandidates,
                                                int* pResults )
{
    if ( nCandidates <= 0 )
        return 1;

    Fxch_Man_t* pMan = pSCHashTable->pFxchMan;
    int nSizeOutputID = pMan->nSizeOutputID;

    if ( nSizeOutputID <= 0 )
    {
        memset( pResults, 0, sizeof(int) * (size_t)nCandidates );
        return 1;
    }

    int totalCandidateInts = 0;
    int i;
    for ( i = 0; i < nCandidates; ++i )
    {
        Vec_Int_t* vCube = Vec_WecEntry( vCubes, pCandidates[i].iCube );
        totalCandidateInts += Vec_IntSize( vCube );
    }

    Vec_Int_t* vRefCube = Vec_WecEntry( vCubes, pReference->iCube );
    int refCubeSize = Vec_IntSize( vRefCube );

    int* hCandidateCubeData = totalCandidateInts ? ABC_ALLOC( int, totalCandidateInts ) : NULL;
    int* hCandidateCubeOffsets = ABC_ALLOC( int, nCandidates );
    int* hCandidateCubeLengths = ABC_ALLOC( int, nCandidates );
    uint32_t* hCandidateIds = ABC_ALLOC( uint32_t, nCandidates );
    int* hCandidateILit0 = ABC_ALLOC( int, nCandidates );
    int* hCandidateILit1 = ABC_ALLOC( int, nCandidates );
    int* hCandidateOutputs = ABC_ALLOC( int, nCandidates * nSizeOutputID );
    int* hRefCubeData = refCubeSize ? ABC_ALLOC( int, refCubeSize ) : NULL;
    int* hRefOutputData = ABC_ALLOC( int, nSizeOutputID );

    if ( ( totalCandidateInts && hCandidateCubeData == NULL ) ||
         hCandidateCubeOffsets == NULL || hCandidateCubeLengths == NULL ||
         hCandidateIds == NULL || hCandidateILit0 == NULL ||
         hCandidateILit1 == NULL || hCandidateOutputs == NULL ||
         ( refCubeSize && hRefCubeData == NULL ) ||
         hRefOutputData == NULL )
    {
        if ( hCandidateCubeData ) ABC_FREE( hCandidateCubeData );
        ABC_FREE( hCandidateCubeOffsets );
        ABC_FREE( hCandidateCubeLengths );
        ABC_FREE( hCandidateIds );
        ABC_FREE( hCandidateILit0 );
        ABC_FREE( hCandidateILit1 );
        ABC_FREE( hCandidateOutputs );
        if ( hRefCubeData ) ABC_FREE( hRefCubeData );
        ABC_FREE( hRefOutputData );
        return 0;
    }

    int offset = 0;
    for ( i = 0; i < nCandidates; ++i )
    {
        Vec_Int_t* vCube = Vec_WecEntry( vCubes, pCandidates[i].iCube );
        int len = Vec_IntSize( vCube );
        hCandidateCubeOffsets[i] = offset;
        hCandidateCubeLengths[i] = len;
        if ( len > 0 )
        {
            memcpy( hCandidateCubeData + offset, vCube->pArray, sizeof(int) * (size_t)len );
        }
        offset += len;

        hCandidateIds[i] = pCandidates[i].Id;
        hCandidateILit0[i] = (int)pCandidates[i].iLit0;
        hCandidateILit1[i] = (int)pCandidates[i].iLit1;

        int* pOutput = Vec_IntEntryP( pMan->vOutputID, pCandidates[i].iCube * nSizeOutputID );
        memcpy( hCandidateOutputs + i * nSizeOutputID, pOutput, sizeof(int) * (size_t)nSizeOutputID );
    }

    if ( refCubeSize > 0 )
        memcpy( hRefCubeData, vRefCube->pArray, sizeof(int) * (size_t)refCubeSize );

    int* pRefOutput = Vec_IntEntryP( pMan->vOutputID, pReference->iCube * nSizeOutputID );
    memcpy( hRefOutputData, pRefOutput, sizeof(int) * (size_t)nSizeOutputID );

    int *dCandidateCubeData = NULL, *dCandidateCubeOffsets = NULL, *dCandidateCubeLengths = NULL;
    uint32_t* dCandidateIds = NULL;
    int *dCandidateILit0 = NULL, *dCandidateILit1 = NULL;
    int* dCandidateOutputs = NULL;
    int* dRefCubeData = NULL;
    int* dRefOutputData = NULL;
    int* dResults = NULL;

    cudaError_t status = cudaSuccess;

    if ( totalCandidateInts )
    {
        status = cudaMalloc( (void**)&dCandidateCubeData, sizeof(int) * (size_t)totalCandidateInts );
        if ( status != cudaSuccess ) goto gpu_cleanup;
        status = cudaMemcpy( dCandidateCubeData, hCandidateCubeData, sizeof(int) * (size_t)totalCandidateInts, cudaMemcpyHostToDevice );
        if ( status != cudaSuccess ) goto gpu_cleanup;
    }

    status = cudaMalloc( (void**)&dCandidateCubeOffsets, sizeof(int) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateCubeOffsets, hCandidateCubeOffsets, sizeof(int) * (size_t)nCandidates, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dCandidateCubeLengths, sizeof(int) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateCubeLengths, hCandidateCubeLengths, sizeof(int) * (size_t)nCandidates, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dCandidateIds, sizeof(uint32_t) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateIds, hCandidateIds, sizeof(uint32_t) * (size_t)nCandidates, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dCandidateILit0, sizeof(int) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateILit0, hCandidateILit0, sizeof(int) * (size_t)nCandidates, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dCandidateILit1, sizeof(int) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateILit1, hCandidateILit1, sizeof(int) * (size_t)nCandidates, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dCandidateOutputs, sizeof(int) * (size_t)(nCandidates * nSizeOutputID) );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dCandidateOutputs, hCandidateOutputs, sizeof(int) * (size_t)(nCandidates * nSizeOutputID), cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    if ( refCubeSize )
    {
        status = cudaMalloc( (void**)&dRefCubeData, sizeof(int) * (size_t)refCubeSize );
        if ( status != cudaSuccess ) goto gpu_cleanup;
        status = cudaMemcpy( dRefCubeData, hRefCubeData, sizeof(int) * (size_t)refCubeSize, cudaMemcpyHostToDevice );
        if ( status != cudaSuccess ) goto gpu_cleanup;
    }

    status = cudaMalloc( (void**)&dRefOutputData, sizeof(int) * (size_t)nSizeOutputID );
    if ( status != cudaSuccess ) goto gpu_cleanup;
    status = cudaMemcpy( dRefOutputData, hRefOutputData, sizeof(int) * (size_t)nSizeOutputID, cudaMemcpyHostToDevice );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMalloc( (void**)&dResults, sizeof(int) * (size_t)nCandidates );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = FxchCudaLaunchCompareKernel( dCandidateCubeData,
                                          dCandidateCubeOffsets,
                                          dCandidateCubeLengths,
                                          dCandidateIds,
                                          dCandidateILit0,
                                          dCandidateILit1,
                                          dCandidateOutputs,
                                          nCandidates,
                                          nSizeOutputID,
                                          dRefCubeData,
                                          refCubeSize,
                                          pReference->Id,
                                          (int)pReference->iLit0,
                                          (int)pReference->iLit1,
                                          dRefOutputData,
                                          dResults );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    status = cudaMemcpy( pResults, dResults, sizeof(int) * (size_t)nCandidates, cudaMemcpyDeviceToHost );
    if ( status != cudaSuccess ) goto gpu_cleanup;

    // Success path
    if ( hCandidateCubeData ) ABC_FREE( hCandidateCubeData );
    ABC_FREE( hCandidateCubeOffsets );
    ABC_FREE( hCandidateCubeLengths );
    ABC_FREE( hCandidateIds );
    ABC_FREE( hCandidateILit0 );
    ABC_FREE( hCandidateILit1 );
    ABC_FREE( hCandidateOutputs );
    if ( hRefCubeData ) ABC_FREE( hRefCubeData );
    ABC_FREE( hRefOutputData );

    if ( dCandidateCubeData ) cudaFree( dCandidateCubeData );
    cudaFree( dCandidateCubeOffsets );
    cudaFree( dCandidateCubeLengths );
    cudaFree( dCandidateIds );
    cudaFree( dCandidateILit0 );
    cudaFree( dCandidateILit1 );
    cudaFree( dCandidateOutputs );
    if ( dRefCubeData ) cudaFree( dRefCubeData );
    cudaFree( dRefOutputData );
    cudaFree( dResults );

    return 1;

gpu_cleanup:
    if ( hCandidateCubeData ) ABC_FREE( hCandidateCubeData );
    ABC_FREE( hCandidateCubeOffsets );
    ABC_FREE( hCandidateCubeLengths );
    ABC_FREE( hCandidateIds );
    ABC_FREE( hCandidateILit0 );
    ABC_FREE( hCandidateILit1 );
    ABC_FREE( hCandidateOutputs );
    if ( hRefCubeData ) ABC_FREE( hRefCubeData );
    ABC_FREE( hRefOutputData );

    if ( dCandidateCubeData ) cudaFree( dCandidateCubeData );
    if ( dCandidateCubeOffsets ) cudaFree( dCandidateCubeOffsets );
    if ( dCandidateCubeLengths ) cudaFree( dCandidateCubeLengths );
    if ( dCandidateIds ) cudaFree( dCandidateIds );
    if ( dCandidateILit0 ) cudaFree( dCandidateILit0 );
    if ( dCandidateILit1 ) cudaFree( dCandidateILit1 );
    if ( dCandidateOutputs ) cudaFree( dCandidateOutputs );
    if ( dRefCubeData ) cudaFree( dRefCubeData );
    if ( dRefOutputData ) cudaFree( dRefOutputData );
    if ( dResults ) cudaFree( dResults );

    return 0;
}

static void FxchCuda_SCHashTableCompareBatch( Fxch_SCHashTable_t* pSCHashTable,
                                               Vec_Wec_t* vCubes,
                                               Fxch_SubCube_t* pReference,
                                               Fxch_SubCube_t* pCandidates,
                                               int nCandidates,
                                               int* pResults )
{
    if ( nCandidates <= 0 )
        return;

    memset( pResults, 0, sizeof(int) * (size_t)nCandidates );

    if ( FxchCuda_SCHashTableCompareBatchGpu( pSCHashTable, vCubes, pReference, pCandidates, nCandidates, pResults ) )
        return;

    // GPU failed – fallback to CPU sequential comparisons
    int i;
    for ( i = 0; i < nCandidates; ++i )
        pResults[i] = FxchCuda_SCHashTableEntryCompareCPU( pSCHashTable, vCubes, &pCandidates[i], pReference );
}

// -----------------------------------------------------------------------------
// Public wrappers
// -----------------------------------------------------------------------------

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, short int usingGpu )
{
    if ( !usingGpu )
        return Fxch_SCHashTableCreate( pFxchMan, nEntries );

    return Fxch_SCHashTableCreate( pFxchMan, nEntries );
}

void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, short int usingGpu )
{
    if ( !usingGpu )
    {
        Fxch_SCHashTableDelete( pSCHashTable );
        return;
    }

    Fxch_SCHashTableDelete( pSCHashTable );
}

int FxchCuda_SCHashTableInsert( Fxch_SCHashTable_t* pSCHashTable,
                                Vec_Wec_t* vCubes,
                                uint32_t SubCubeID,
                                uint32_t iCube,
                                uint32_t iLit0,
                                uint32_t iLit1,
                                char fUpdate,
                                short int usingGpu )
{
    if ( !usingGpu )
        return Fxch_SCHashTableInsert( pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate );

    int iNewEntry;
    int Pairs = 0;
    uint32_t BinID;
    Fxch_SCHashTable_Entry_t* pBin;
    Fxch_SubCube_t* pNewEntry;
    int iEntry;

    MurmurHash3_x86_32( ( void* ) &SubCubeID, sizeof( int ), 0x9747b28c, &BinID );
    pBin = pSCHashTable->pBins + ( BinID & pSCHashTable->SizeMask );

    if ( pBin->vSCData == NULL )
    {
        pBin->vSCData = ABC_CALLOC( Fxch_SubCube_t, 16 );
        pBin->Size = 0;
        pBin->Cap = 16;
    }
    else if ( pBin->Size == pBin->Cap )
    {
        assert( pBin->Cap <= 0xAAAA );
        pBin->Cap = ( pBin->Cap >> 1 ) * 3;
        pBin->vSCData = ABC_REALLOC( Fxch_SubCube_t, pBin->vSCData, pBin->Cap );
    }

    iNewEntry = pBin->Size++;
    pBin->vSCData[iNewEntry].Id = SubCubeID;
    pBin->vSCData[iNewEntry].iCube = iCube;
    pBin->vSCData[iNewEntry].iLit0 = iLit0;
    pBin->vSCData[iNewEntry].iLit1 = iLit1;
    pSCHashTable->nEntries++;

    if ( pBin->Size == 1 )
        return 0;

    pNewEntry = &( pBin->vSCData[iNewEntry] );

    int nCandidates = (int)pBin->Size - 1;
    int haveBatchResults = 0;
    int* pCompareResults = NULL;

    if ( nCandidates > 0 )
    {
        int validCount = 0;
        int idxTemp;
        for ( idxTemp = 0; idxTemp < nCandidates; ++idxTemp )
        {
            Fxch_SubCube_t* pEntry = &( pBin->vSCData[idxTemp] );
            if ( ( pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0 ) ||
                 ( pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0 ) )
                continue;
            validCount++;
        }

        if ( validCount > 0 )
        {
            Fxch_SubCube_t* pCandidateSubset = ABC_ALLOC( Fxch_SubCube_t, validCount );
            int* pIndexMap = ABC_ALLOC( int, validCount );
            int* pSubsetResults = ABC_ALLOC( int, validCount );

            if ( pCandidateSubset && pIndexMap && pSubsetResults )
            {
                int idx = 0;
                for ( idxTemp = 0; idxTemp < nCandidates; ++idxTemp )
                {
                    Fxch_SubCube_t* pEntry = &( pBin->vSCData[idxTemp] );
                    if ( ( pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0 ) ||
                         ( pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0 ) )
                        continue;
                    pCandidateSubset[idx] = *pEntry;
                    pIndexMap[idx] = idxTemp;
                    idx++;
                }

                FxchCuda_SCHashTableCompareBatch( pSCHashTable, vCubes, pNewEntry, pCandidateSubset, validCount, pSubsetResults );

                pCompareResults = ABC_CALLOC( int, nCandidates );
                if ( pCompareResults )
                {
                    int k;
                    for ( k = 0; k < validCount; ++k )
                        pCompareResults[ pIndexMap[k] ] = pSubsetResults[k];
                    haveBatchResults = 1;
                }

            }

            if ( pCandidateSubset ) ABC_FREE( pCandidateSubset );
            if ( pIndexMap ) ABC_FREE( pIndexMap );
            if ( pSubsetResults ) ABC_FREE( pSubsetResults );
        }
    }

    for ( iEntry = 0; iEntry < (int)pBin->Size - 1; iEntry++ )
    {
        Fxch_SubCube_t* pEntry = &( pBin->vSCData[iEntry] );
        int* pOutputID0 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
        int* pOutputID1 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pNewEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
        int Result = 0;
        int Base;
        int iNewDiv = -1, i, z;

        if ( ( pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0 ) ||
             ( pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0 ) )
            continue;

        if ( haveBatchResults )
        {
            if ( !pCompareResults || !pCompareResults[iEntry] )
                continue;
        }
        else
        {
            if ( !FxchCuda_SCHashTableEntryCompareCPU( pSCHashTable, vCubes, pEntry, pNewEntry ) )
                continue;
        }

        if ( ( pEntry->iLit0 == 0 ) || ( pNewEntry->iLit0 == 0 ) )
        {
            Vec_Int_t* vCube0 = Fxch_ManGetCube( pSCHashTable->pFxchMan, pEntry->iCube );
            Vec_Int_t* vCube1 = Fxch_ManGetCube( pSCHashTable->pFxchMan, pNewEntry->iCube );

            if ( Vec_IntSize( vCube0 ) > Vec_IntSize( vCube1 ) )
            {
                Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pEntry->iCube );
                Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube );
            }
            else
            {
                Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube );
                Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pEntry->iCube );
            }

            continue;
        }

        Base = Fxch_DivCreate( pSCHashTable->pFxchMan, pEntry, pNewEntry );

        if ( Base < 0 )
            continue;

        for ( i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID; i++ )
            Result += Fxch_CountOnes( pOutputID0[i] & pOutputID1[i] );

        for ( z = 0; z < Result; z++ )
            iNewDiv = Fxch_DivAdd( pSCHashTable->pFxchMan, fUpdate, 0, Base );

        Vec_WecPush( pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pEntry->iCube );
        Vec_WecPush( pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pNewEntry->iCube );

        Pairs++;
    }

    if ( pCompareResults ) ABC_FREE( pCompareResults );

    return Pairs;
}

int FxchCuda_SCHashTableRemove( Fxch_SCHashTable_t* pSCHashTable,
                                Vec_Wec_t* vCubes,
                                uint32_t SubCubeID,
                                uint32_t iCube,
                                uint32_t iLit0,
                                uint32_t iLit1,
                                char fUpdate,
                                short int usingGpu )
{
    if ( !usingGpu )
        return Fxch_SCHashTableRemove( pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate );

    int iEntry;
    int Pairs = 0;
    uint32_t BinID;
    Fxch_SCHashTable_Entry_t* pBin;
    Fxch_SubCube_t* pEntry;
    int idx;

    MurmurHash3_x86_32( ( void* ) &SubCubeID, sizeof( int ), 0x9747b28c, &BinID );

    pBin = pSCHashTable->pBins + ( BinID & pSCHashTable->SizeMask );

    if ( pBin->Size == 1 )
    {
        pBin->Size = 0;
        return 0;
    }

    for ( iEntry = 0; iEntry < (int)pBin->Size; iEntry++ )
        if ( pBin->vSCData[iEntry].iCube == iCube )
            break;

    assert( ( iEntry != (int)pBin->Size ) && ( pBin->Size != 0 ) );

    pEntry = &( pBin->vSCData[iEntry] );

    int nCandidates = (int)pBin->Size - 1;
    int haveBatchResults = 0;
    int* pCompareResults = NULL;

    if ( nCandidates > 0 )
    {
        int validCount = 0;
        for ( idx = 0; idx < (int)pBin->Size; idx++ )
            if ( idx != iEntry )
            {
                Fxch_SubCube_t* pNextEntry = &( pBin->vSCData[idx] );
                if ( ( pEntry->iLit1 != 0 && pNextEntry->iLit1 == 0 ) ||
                     ( pEntry->iLit1 == 0 && pNextEntry->iLit1 != 0 ) )
                    continue;
                validCount++;
            }

        if ( validCount > 0 )
        {
            Fxch_SubCube_t* pCandidateSubset = ABC_ALLOC( Fxch_SubCube_t, validCount );
            int* pIndexMap = ABC_ALLOC( int, validCount );
            int* pSubsetResults = ABC_ALLOC( int, validCount );

            if ( pCandidateSubset && pIndexMap && pSubsetResults )
            {
                int idy = 0;
                for ( idx = 0; idx < (int)pBin->Size; idx++ )
                    if ( idx != iEntry )
                    {
                        Fxch_SubCube_t* pNextEntry = &( pBin->vSCData[idx] );
                        if ( ( pEntry->iLit1 != 0 && pNextEntry->iLit1 == 0 ) ||
                             ( pEntry->iLit1 == 0 && pNextEntry->iLit1 != 0 ) )
                            continue;
                        pCandidateSubset[idy] = *pNextEntry;
                        pIndexMap[idy] = idx;
                        idy++;
                    }

                FxchCuda_SCHashTableCompareBatch( pSCHashTable, vCubes, pEntry, pCandidateSubset, validCount, pSubsetResults );

                pCompareResults = ABC_CALLOC( int, nCandidates );
                if ( pCompareResults )
                {
                    int k;
                    for ( k = 0; k < validCount; ++k )
                    {
                        int actualIdx = pIndexMap[k];
                        int mappedIdx = actualIdx < iEntry ? actualIdx : actualIdx - 1;
                        if ( mappedIdx >= 0 && mappedIdx < nCandidates )
                            pCompareResults[mappedIdx] = pSubsetResults[k];
                    }
                    haveBatchResults = 1;
                }
            }

            if ( pCandidateSubset ) ABC_FREE( pCandidateSubset );
            if ( pIndexMap ) ABC_FREE( pIndexMap );
            if ( pSubsetResults ) ABC_FREE( pSubsetResults );
        }
    }

    for ( idx = 0; idx < (int)pBin->Size; idx++ )
    if ( idx != iEntry )
    {
        int Base, iDiv = -1;
        int i, z, iCube0, iCube1;
        Fxch_SubCube_t* pNextEntry = &( pBin->vSCData[idx] );
        Vec_Int_t* vDivCubePairs;
        int* pOutputID0 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
        int* pOutputID1 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, pNextEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
        int Result = 0;

        if ( ( pEntry->iLit1 != 0 && pNextEntry->iLit1 == 0 ) ||
             ( pEntry->iLit1 == 0 && pNextEntry->iLit1 != 0 ) )
            continue;

        int mappedIdx = idx < iEntry ? idx : idx - 1;

        if ( haveBatchResults )
        {
            if ( mappedIdx < 0 || mappedIdx >= nCandidates || !pCompareResults || !pCompareResults[mappedIdx] )
                continue;
        }
        else
        {
            if ( !FxchCuda_SCHashTableEntryCompareCPU( pSCHashTable, vCubes, pEntry, pNextEntry ) )
                continue;
        }

        if ( pEntry->iLit0 == 0 || pNextEntry->iLit0 == 0 )
            continue;

        Base = Fxch_DivCreate( pSCHashTable->pFxchMan, pNextEntry, pEntry );

        if ( Base < 0 )
            continue;

        for ( i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID; i++ )
            Result += Fxch_CountOnes( pOutputID0[i] & pOutputID1[i] );

        for ( z = 0; z < Result; z++ )
            iDiv = Fxch_DivRemove( pSCHashTable->pFxchMan, fUpdate, 0, Base );

        vDivCubePairs = Vec_WecEntry( pSCHashTable->pFxchMan->vDivCubePairs, iDiv );
        Vec_IntForEachEntryDouble( vDivCubePairs, iCube0, iCube1, i )
            if ( ( iCube0 == (int)pNextEntry->iCube && iCube1 == (int)pEntry->iCube )  ||
                 ( iCube0 == (int)pEntry->iCube && iCube1 == (int)pNextEntry->iCube ) )
            {
                Vec_IntDrop( vDivCubePairs, i+1 );
                Vec_IntDrop( vDivCubePairs, i );
            }
        if ( Vec_IntSize( vDivCubePairs ) == 0 )
            Vec_IntErase( vDivCubePairs );

        Pairs++;
    }

    if ( pCompareResults ) ABC_FREE( pCompareResults );

    memmove( pBin->vSCData + iEntry, pBin->vSCData + iEntry + 1, ( pBin->Size - iEntry - 1 ) * sizeof(*pBin->vSCData) );
    pBin->Size -= 1;

    return Pairs;
}

unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, short int usingGpu )
{
    if ( !usingGpu )
        return Fxch_SCHashTableMemory( pHashTable );

    return Fxch_SCHashTableMemory( pHashTable );
}

void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, short int usingGpu )
{
    if ( !usingGpu )
    {
        Fxch_SCHashTablePrint( pHashTable );
        return;
    }

    Fxch_SCHashTablePrint( pHashTable );
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END

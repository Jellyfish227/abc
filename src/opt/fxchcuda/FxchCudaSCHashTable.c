/*


    ASSUME DEPRECATED


*/






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

#if (__GNUC__ >= 8)
  #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

// These are all wrappers with fallbacks to the original fxch implementation

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate(
    Fxch_Man_t* pFxchMan, 
    int nEntries, 
    short int usingGpu)
{
    if (!usingGpu) {
        return Fxch_SCHashTableCreate(pFxchMan, nEntries);
    }
    
    // GPU path: Just allocate empty structure
    Fxch_SCHashTable_t* pSCHashTable = ABC_CALLOC(Fxch_SCHashTable_t, 1);
    pSCHashTable->pFxchMan = pFxchMan;
    pSCHashTable->nEntries = nEntries;
    pSCHashTable->usingGpu = 1;
    
    // Allocate GPU structure (but don't transfer data yet!)
    Fxch_SCHashTable_GPU_t* d_table = ABC_CALLOC(Fxch_SCHashTable_GPU_t, 1);
    d_table->nEntries = nEntries;
    
    // Initialize hash table
    size_t estimatedCapacity = (size_t)(nEntries / 0.7 * 1.2);
    d_table->d_subcubeMap = new cuco::static_multimap<uint32_t, Fxch_SubCube_t>(
        estimatedCapacity,
        cuco::empty_key<uint32_t>{UINT32_MAX},
        cuco::empty_value<Fxch_SubCube_t>{{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}}
    );
    
    pSCHashTable->d_gpu_table = (void*)d_table;
    
    return pSCHashTable;
}


void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTableDelete(pSCHashTable);
        return;
    }

    // GPU cleanup
    Fxch_SCHashTable_GPU_t* d_table = (Fxch_SCHashTable_GPU_t*)pSCHashTable->d_gpu_table;
    FxchCuda_SCHashTableDeleteGPU(d_table);
    free(pSCHashTable);

}

// int FxchCuda_SCHashTableInsert( Fxch_SCHashTable_t* pSCHashTable,
//                             Vec_Wec_t* vCubes,
//                             char fUpdate,
//                             short int usingGpu )
// {



// }


int FxchCuda_SCHashTableRemove( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            char fUpdate,
                            short int usingGpu )
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }

    Fxch_SCHashTable_GPU_t* d_table = 
        (Fxch_SCHashTable_GPU_t*)pSCHashTable->d_gpu_table;
    FxchCuda_SCHashTableRemoveGPU(d_table, SubCubeID, iCube, iLit0, iLit1);
    return 1;
}


unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, short int usingGpu) 
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableMemory(pHashTable);
    }
        
    Fxch_SCHashTable_GPU_t* d_table = 
        (Fxch_SCHashTable_GPU_t*)pHashTable->d_gpu_table;
    return FxchCuda_SCHashTableMemoryGPU(d_table);
}


void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTablePrint(pHashTable);
        return;
    }
     
    Fxch_SCHashTable_GPU_t* d_table = 
        (Fxch_SCHashTable_GPU_t*)pHashTable->d_gpu_table;
    FxchCuda_SCHashTablePrintGPU(d_table);
};

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END

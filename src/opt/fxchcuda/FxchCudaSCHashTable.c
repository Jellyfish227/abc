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

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, short int usingGpu )
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableCreate(pFxchMan, nEntries);
    }
    
    Vec_Wec_t* tbsCubes = pFxchMan->vCubes; // To Be Serialzied Cubes

    int totalElements = Vec_WecSizeSize(tbsCubes);
    int numCubes = Vec_WecSize(tbsCubes);

    /*  
        For Jellyfish: 
        Today I learned that I can create arrays on the heap with malloc, 
        I've never done this before so make sure this is right before proceeding
    */

    int* flatData = ABC_ALLOC(int, totalElements); // array that stores every literal
    int* levelSizes = ABC_ALLOC(int, numCubes); // array that stores number of literals for each cube

    int offset = 0;
    Vec_Int_t* vLevel;
    int i;
    
    Vec_WecForEachLevel(tbsCubes, vLevel, i) {
        int levelSize = Vec_IntSize(vLevel);
        levelSizes[i] = levelSize;
        memcpy(flatData + offset, Vec_IntArray(vLevel), levelSize * sizeof(int)); // copies each cube into the flat array 
        offset += levelSize;
    }

    // Call GPU function (defined in .cu file)
    Fxch_SCHashTable_t* result = FxchCuda_TransferAndAllocateGPU(
        flatData, levelSizes, totalElements, numCubes, nEntries
    );
    
    // I will free it, we are not developing escape from tarkov :)
    free(flatData);
    free(levelSizes);
    return result;
}


void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTableDelete(pSCHashTable);
        return;
    }
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
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
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
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }

}


unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, short int usingGpu) 
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableMemory(pHashTable);
    }
}


void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTablePrint(pHashTable);
        return;
    }
};

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END

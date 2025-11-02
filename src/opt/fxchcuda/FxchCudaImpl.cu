#include "FxchCuda.h"

Fxch_SCHashTable_t* FxchCuda_TransferAndAllocateGPU(int *flatData, 
                                                    int *levelSizes,
                                                    int totalElements,
                                                    int numLevels,
                                                    int nEntries ) 
{
    /*
        For jellyfish:

        flatData: pointer to an array of every literal
        levelSizes: pointer to an array of # of literals per cube

        e.g.
        flatData: [0,2,3,5,7]
        levelSizes: [2,3]

        the actual expression = 02 + 357
    */

    /* to be implemented*/
};

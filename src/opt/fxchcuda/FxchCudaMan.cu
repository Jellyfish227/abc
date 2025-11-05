/*

    THIS FILE ONLY CONTAINS THE ENTRY POINT

*/

#include "FxchCuda.h"

// this function :
// 1. Extracts all subcubes into flat arrays
// 2. creates the global static_multimap
// 3. Calls the batch insertion & comparison
// 4. Updates the CPU hash table
void FxchCuda_ManDivDoubleCube( Fxch_Man_t* pFxchMan, int fAdd, int fUpdate )
{
    // Step 1: Generate ALL subcubes on CPU (for double-cube divisors only)
    int totalSubcubes = 0;
    Vec_Int_t* vCube;
    int iCube;
    
    // Count subcubes
    Vec_WecForEachLevel(FxchMan->vCubes, vCube, iCube) {
        int nLits = Vec_IntSize(vCube) - 1;
        if (nLits < 2) continue;
        // 0 lits removed: 1
        // 1 lit removed: nLits
        // 2 lits removed: (nLits choose 2)
        totalSubcubes += 1 + nLits + (nLits * (nLits - 1)) / 2;
    }
    
    uint32_t* h_subcubeIds = ABC_ALLOC(uint32_t, totalSubcubes);
    uint32_t* h_cubeIndices = ABC_ALLOC(uint32_t, totalSubcubes);
    uint32_t* h_iLit0 = ABC_ALLOC(uint32_t, totalSubcubes);
    uint32_t* h_iLit1 = ABC_ALLOC(uint32_t, totalSubcubes);
    
    int subcubeIdx = 0;
    
    // Generate subcubes (same logic as Fxch_ManDivDoubleCube)
    Vec_WecForEachLevel(FxchMan->vCubes, vCube, iCube) {
        int nLits = Vec_IntSize(vCube) - 1;
        if (nLits < 2) continue;
        
        Vec_Int_t* vLitHashKeys = pFxchMan->vLitHashKeys;
        uint32_t SubCubeID = 0;
        int iLit0, Lit0;
        
        // Compute base hash
        Vec_IntForEachEntryStart(vCube, Lit0, iLit0, 1)
            SubCubeID += Vec_IntEntry(vLitHashKeys, Lit0);
        
        // 0 literals removed
        h_subcubeIds[subcubeIdx] = SubCubeID;
        h_cubeIndices[subcubeIdx] = iCube;
        h_iLit0[subcubeIdx] = 0;
        h_iLit1[subcubeIdx] = 0;
        subcubeIdx++;
        
        // 1 literal removed
        Vec_IntForEachEntryStart(vCube, Lit0, iLit0, 1) {
            uint32_t temp = SubCubeID - Vec_IntEntry(vLitHashKeys, Lit0);
            h_subcubeIds[subcubeIdx] = temp;
            h_cubeIndices[subcubeIdx] = iCube;
            h_iLit0[subcubeIdx] = iLit0;
            h_iLit1[subcubeIdx] = 0;
            subcubeIdx++;
        }
        
        // 2 literals removed
        Vec_IntForEachEntryStart(vCube, Lit0, iLit0, 1) {
            uint32_t temp1 = SubCubeID - Vec_IntEntry(vLitHashKeys, Lit0);
            int Lit1, iLit1;
            Vec_IntForEachEntryStart(vCube, Lit1, iLit1, iLit0 + 1) {
                uint32_t temp2 = temp1 - Vec_IntEntry(vLitHashKeys, Lit1);
                h_subcubeIds[subcubeIdx] = temp2;
                h_cubeIndices[subcubeIdx] = iCube;
                h_iLit0[subcubeIdx] = iLit0;
                h_iLit1[subcubeIdx] = iLit1;
                subcubeIdx++;
            }
        }
    }
    
    // Step 2: Initialize/resize GPU hash table based on subcube count
    Fxch_SCHashTable_GPU_t* d_table = 
        (Fxch_SCHashTable_GPU_t*)pFxchMan->pSCHashTable->d_gpu_table;
    
    // Check if we need to (re)allocate the hash table
    // In practice, this code will only be run ONCE, this code is pretty meaningless
    if (d_table->d_subcubeMap == NULL || d_table->capacity < totalSubcubes) {
        
        // Free old hash table if it exists
        if (d_table->d_subcubeMap != NULL) {
            delete d_table->d_subcubeMap;
        }
        
        // Calculate new capacity with overhead for hash collisions
        // Use 70% load factor: capacity = totalSubcubes / 0.7
        // Add 20% extra buffer: capacity *= 1.2
        size_t estimatedCapacity = static_cast<size_t>(totalSubcubes / 0.7 * 1.2);
        
        // Create multimap with MurmurHash3-based hasher
        d_table->d_subcubeMap = new cuco::static_multimap<
            uint32_t,                     // Key type
            Fxch_SubCube_t,              // Value type
            cuda::thread_scope_device,    // Scope
            MurmurHash3Hasher,           // Custom hash function ← IMPORTANT!
            thrust::equal_to<uint32_t>   // Key equality
        >(
            estimatedCapacity,
            cuco::empty_key<uint32_t>{UINT32_MAX},
            cuco::empty_value<Fxch_SubCube_t>{{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}},
            {},                          // Allocator (default)
            {},                          // Stream (default)
            MurmurHash3Hasher{}          // Hash function instance
        );
        
        d_table->capacity = estimatedCapacity;
        d_table->nSubcubes = totalSubcubes;
        
        printf("GPU: Allocated hash table for %d subcubes (capacity: %zu)\n", 
               totalSubcubes, estimatedCapacity);
    }
    
    // Step 3: GPU batch insert
    FxchCuda_InsertSubcubesBatch(d_table, h_subcubeIds, h_cubeIndices,
                                  h_iLit0, h_iLit1, totalSubcubes);
    
    // Step 4: GPU batch compare
    int* d_matching_pairs;
    int numMatches = FxchCuda_CompareSubcubesBatch(
        d_table, h_subcubeIds, h_cubeIndices, h_iLit0, h_iLit1,
        totalSubcubes, &d_matching_pairs);
    
    // Step 6: Copy matches back, create divisors on CPU
    int* h_matching_pairs = ABC_ALLOC(int, numMatches * 2);
    cudaMemcpy(h_matching_pairs, d_matching_pairs, 
               numMatches * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numMatches; i++) {
        // Use existing divisor creation logic
        // (same as what happens in Fxch_ManSCAddRemove when match found)
    }
    
    // Cleanup
    ABC_FREE(flatData);
    ABC_FREE(levelSizes);
    ABC_FREE(h_subcubeIds);
    ABC_FREE(h_cubeIndices);
    ABC_FREE(h_iLit0);
    ABC_FREE(h_iLit1);
    ABC_FREE(h_matching_pairs);
    cudaFree(d_matching_pairs);
}

// In FxchCuda.cu - GPU version of MurmurHash3
__device__ __forceinline__ uint32_t murmur3_fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

__device__ __forceinline__ uint32_t murmur3_rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

__device__ uint32_t MurmurHash3_x86_32_device(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    
    uint32_t h1 = seed;
    
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    // Body
    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        
        k1 *= c1;
        k1 = murmur3_rotl32(k1, 15);
        k1 *= c2;
        
        h1 ^= k1;
        h1 = murmur3_rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }
    
    // Tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16; // fallthrough
        case 2: k1 ^= tail[1] << 8;  // fallthrough
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = murmur3_rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }
    
    // Finalization
    h1 ^= len;
    return murmur3_fmix32(h1);
}

// Custom hasher that uses MurmurHash3 (same as CPU)
struct MurmurHash3Hasher {
    __device__ uint32_t operator()(uint32_t const& key) const noexcept {
        // The key is already a hash (SubCubeID), but we need to apply
        // MurmurHash3's final mixing to ensure good distribution
        return murmur3_fmix32(key);
    }
};
#include "FxchBackend.h"

static Fxch_Backend_t* g_pBackend = NULL;

Fxch_Backend_t* Fxch_GetBackend()
{
    if (g_pBackend == NULL)
    {
#if FXCH_USE_CUDA
        extern Fxch_Backend_t* Fxch_GetBackendGPU();
        g_pBackend = Fxch_GetBackendGPU();
#else
        extern Fxch_Backend_t* Fxch_GetBackendCPU();
        g_pBackend = Fxch_GetBackendCPU();
#endif
    }
    return g_pBackend;
}
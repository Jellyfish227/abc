#ifndef ABC__opt__fxch__backend_h
#define ABC__opt__fxch__backend_h

#include "Fxch.h"
#include "FxchConfig.h"

typedef struct Fxch_Backend_t_
{
    Fxch_SCHashTable_t* (*SCHashTableCreate)(Fxch_Man_t*, int);
    void (*SCHashTableDelete)(Fxch_SCHashTable_t*);
    int (*SCHashTableInsert)(Fxch_SCHashTable_t*, Vec_Wec_t*, uint32_t, uint32_t, uint32_t, uint32_t, char);
    int (*SCHashTableRemove)(Fxch_SCHashTable_t*, Vec_Wec_t*, uint32_t, uint32_t, uint32_t, uint32_t, char);
} Fxch_Backend_t;

Fxch_Backend_t* Fxch_GetBackend();

#endif
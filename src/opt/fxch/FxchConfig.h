#ifndef ABC__opt__fxch__config_h
#define ABC__opt__fxch__config_h

/* Auto-detect or manual override for CUDA support */
#ifdef ABC_USE_CUDA
  #define FXCH_USE_CUDA 1
#else
  #define FXCH_USE_CUDA 0
#endif

#endif
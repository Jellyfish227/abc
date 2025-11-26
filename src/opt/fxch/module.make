SRC += src/opt/fxch/Fxch.c \
       src/opt/fxch/FxchDiv.c \
       src/opt/fxch/FxchMan.c \
       src/opt/fxch/FxchSCHashTable.c \
       src/opt/fxch/FxchBackendCPU.c \
       src/opt/fxch/FxchBackendFactory.c

ifdef ABC_USE_CUDA
  SRC += src/opt/fxch/FxchBackendGPU.c
endif
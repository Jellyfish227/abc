SRC +=  src/opt/fxch/Fxch.c \
    src/opt/fxch/FxchDiv.c \
    src/opt/fxch/FxchMan.c  \
    src/opt/fxch/FxchSCHashTable.c

# if nvcc in environment, discover fxch-cuda
NVCC := $(shell which nvcc 2>/dev/null)
ifdef NVCC
    ABC_USE_CUDA ?= 1
    $(info $(MSG_PREFIX)Found NVCC at $(NVCC) - enabling CUDA support)
    include $(ABCSRC)/src/opt/fxchcuda/module.make 
else
    ABC_USE_CUDA ?= 0
    $(info $(MSG_PREFIX)NVCC not found - building CPU-only)
endif
SRC +=  src/opt/fxch/Fxch.c \
    src/opt/fxch/FxchDiv.c \
    src/opt/fxch/FxchMan.c  \
    src/opt/fxch/FxchSCHashTable.c

# if nvcc in environment, discover fxch-cuda
NVCC := $(shell which nvcc 2>/dev/null)
ifdef NVCC
    CFLAGS += -DABC_USE_CUDA    # Pass ABC_USE_CUDA to FxchMan to branch func call
    $(call abc_info,$(MSG_PREFIX)Found NVCC at $(NVCC) - enabling CUDA support)
    include $(ABCSRC)/src/opt/fxchcuda/module.make 
else
    $(call abc_info,$(MSG_PREFIX)NVCC not found - building CPU-only)
endif
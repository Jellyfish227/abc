# CUDA linking and compiling initialization 
CUDA_PATH   := $(shell dirname $(shell dirname $(NVCC)))
CUDA_LIBDIR := $(CUDA_PATH)/lib64

INCLUDES    += -I$(CUDA_PATH)/include
LDFLAGS     += -L$(CUDA_LIBDIR)
LIBS        += -lcudart

SRC +=  src/opt/fxchcuda/FxchCudaSCHashTable.c

CUDA_SRC += src/opt/fxchcuda/Kernel.cu

# cuCollection
CUDA_INCLUDE_FLAGS = -Ilib/extern/cuCollections/include

CFLAGS    += -g -pg -no-pie -Wall -Wno-unused-function -Wno-write-strings -Wno-sign-compare $(ARCHFLAGS) -I./lib/readline/include -I./lib/ncurses/include
LDFLAGS	  += -L./lib/readline/lib -lreadline -L./lib/ncurses/lib -lncurses

CXXFLAGS += CUDA_INCLUDE_FLAGS

# Add CUDA objects if CUDA sources exist
CUDA_OBJ := $(patsubst %.cu, %.o, $(CUDA_SRC))
OBJ += $(CUDA_OBJ)

# CUDA compilation rules
%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "$(MSG_PREFIX)\`\` Compiling CUDA:" $(LOCAL_PATH)/$<
	$(VERBOSE)$(NVCC) -c -O3 -arch=sm_60 $(INCLUDES) -ccbin $(CXX) -allow-unsupported-compiler -Xcompiler -fPIC $< -o $@

%.d: %.cu
	@mkdir -p $(dir $@)
	@echo "$(MSG_PREFIX)\`\` Generating dependency:" $(LOCAL_PATH)/$<
	$(VERBOSE)$(ABCSRC)/depends.sh "$(NVCC)" `dirname $*.cu` -O3 -arch=sm_60 $(INCLUDES) -ccbin $(CXX) -allow-unsupported-compiler -Xcompiler -fPIC $< > $@
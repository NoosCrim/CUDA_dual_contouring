OUT := out/
OBJ := out/obj/
DEP := out/dep/

CPP := g++
CPP_FLAGS := -g -Og -std=c++20 -I. -Iexternal -Iinclude

CUDA := nvcc
CUDA_HOME := /opt/cuda/
CUDA_LIB	:= -L$(CUDA_HOME)lib64 -lcudart
CUDA_FLAGS	:= -lineinfo -arch=sm_86 --ptxas-options=-v --use_fast_math --std=c++20 -I. -Iexternal -Iinclude

GL_LIB := -lGL
GLEW_LIB := -lGLEW
GLFW_LIB := -lglfw

.PHONY: run_example example cuda_dc renderer
DEPS := $(shell find $(DEP) -name "*.d" 2>/dev/null)
ifneq ($(DEPS),)
include $(DEPS)
endif

CUDA_DC_OBJ := $(patsubst %.cu,$(OBJ)%.o,$(wildcard cuda_dc/*.cu)) $(patsubst %.cpp,$(OBJ)%.o,$(wildcard cuda_dc/*.cpp))
CUDA_DC_LIB := $(OUT)cuda_dc/cuda_dc.a

cuda_dc: $(CUDA_DC_LIB)
	
$(CUDA_DC_LIB): $(CUDA_DC_OBJ)
	@mkdir -p $(dir $@)
	ar rcs -o $@ $^


EXAMPLE_OBJ := $(patsubst %.cpp,$(OBJ)%.o,$(wildcard example/*.cpp))
EXAMPLE_EXEC := $(OUT)cuda_dc_demo


run_example: $(EXAMPLE_EXEC)
	./$(EXAMPLE_EXEC)

example: $(EXAMPLE_EXEC)

$(EXAMPLE_EXEC): $(EXAMPLE_OBJ) $(CUDA_DC_LIB)
	@mkdir -p $(dir $@)
	g++ -o $@ $^ $(CPP_FLAGS) \
	$(RENDERER_LIB) \
	$(GL_LIB) $(GLEW_LIB) $(GLFW_LIB) \
	$(CUDA_LIB)



renderer:
	@cd renderer && make renderer

$(OBJ)%.o : %.cpp
	@mkdir -p $(dir $(DEP)$*.o) $(dir $@)
	$(CPP) -c $< -o $@ $(CPP_FLAGS) -MMD -MP -MF $(DEP)$*.d

$(OBJ)%.o : %.cu
	@mkdir -p $(dir $(DEP)$*.o) $(dir $@)
	$(CUDA) -c $< -o $@ $(CUDA_FLAGS) -MMD -MP -MF $(DEP)$*.d
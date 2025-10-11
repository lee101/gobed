CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -Wall
CUDA_PATH = /usr/local/cuda
CUVS_PATH = /usr/local

INCLUDES = -I$(CUDA_PATH)/include -I$(CUVS_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUVS_PATH)/lib
LIBS = -lcuvs -lcuvs_c -lcudart -lcublas -lcusolver -lcusparse -lm -lpthread

# Targets
all: test_cagra_c test_cagra_quality test_cagra_exact_match test_cagra_comprehensive test_cagra_simple_comprehensive test_cuvs_working test_cagra_optimization

test_cagra_c: test_cagra_c.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cagra_quality: test_cagra_quality.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cagra_exact_match: test_cagra_exact_match.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cagra_comprehensive: test_cagra_comprehensive.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cagra_simple_comprehensive: test_cagra_simple_comprehensive.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cuvs_working: test_cuvs_working.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

test_cagra_optimization: test_cagra_optimization.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) $(LIBS)

run: test_cagra_c
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_c

run_quality: test_cagra_quality
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_quality

run_exact: test_cagra_exact_match
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_exact_match

run_comprehensive: test_cagra_comprehensive
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_comprehensive

run_simple_comprehensive: test_cagra_simple_comprehensive
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_simple_comprehensive

run_cuvs_working: test_cuvs_working
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cuvs_working

run_optimization: test_cagra_optimization
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(CUVS_PATH)/lib:$$LD_LIBRARY_PATH ./test_cagra_optimization

clean:
	rm -f test_cagra_c test_cagra_quality test_cagra_exact_match test_cagra_comprehensive test_cagra_simple_comprehensive test_cuvs_working test_cagra_optimization *.o

.PHONY: all run run_quality run_exact run_comprehensive run_simple_comprehensive run_cuvs_working run_optimization clean
CXX = g++
MPICXX = mpicxx
NVCC = nvcc

CXXFLAGS = -O3 -Wall -std=c++17 -Isrc
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -std=c++17 -Isrc

TARGETS = tree_serial tree_mpi_omp

# 檢查系統是否有 nvcc，若有則加入 cuda 編譯目標
NVCC_EXISTS := $(shell which $(NVCC) 2>/dev/null)
ifneq ($(NVCC_EXISTS),)
    TARGETS += tree_cuda
endif

all: $(TARGETS)

tree_serial: src/tree_serial.cpp src/common.h
	$(CXX) $(CXXFLAGS) src/tree_serial.cpp -o tree_serial

tree_mpi_omp: src/tree_mpi_omp.cpp src/common.h
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) src/tree_mpi_omp.cpp -o tree_mpi_omp

tree_cuda: src/tree_cuda.cu src/common.h
	$(NVCC) $(NVCCFLAGS) src/tree_cuda.cu -o tree_cuda

clean:
	rm -f tree_serial tree_mpi_omp tree_cuda *.o

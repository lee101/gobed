# Makefile for native LibTorch integration

LIBTORCH_ROOT = /home/lee/code/gobed/libtorch
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O3
INCLUDES = -I$(LIBTORCH_ROOT)/include -I$(LIBTORCH_ROOT)/include/torch/csrc/api/include
LDFLAGS = -L$(LIBTORCH_ROOT)/lib -ltorch -ltorch_cpu -lc10 -shared

WRAPPER_LIB = libtorch_cgo_wrapper.so
WRAPPER_SRC = torch_cgo_wrapper.cpp

.PHONY: all clean test

all: $(WRAPPER_LIB)

$(WRAPPER_LIB): $(WRAPPER_SRC) torch_cgo_wrapper.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "✅ Built $(WRAPPER_LIB)"

clean:
	rm -f $(WRAPPER_LIB)
	@echo "🧹 Cleaned build artifacts"

test: $(WRAPPER_LIB)
	@echo "🧪 Testing native LibTorch integration..."
	LD_LIBRARY_PATH=$(LIBTORCH_ROOT)/lib:. go test -v ./...

install: $(WRAPPER_LIB)
	@echo "📦 Installing shared library..."
	@mkdir -p ../lib
	cp $(WRAPPER_LIB) ../lib/
	@echo "✅ Installed to ../lib/"

help:
	@echo "Available targets:"
	@echo "  all     - Build the LibTorch CGO wrapper"
	@echo "  clean   - Remove build artifacts"
	@echo "  test    - Run tests with the wrapper"
	@echo "  install - Install shared library"
	@echo "  help    - Show this help"
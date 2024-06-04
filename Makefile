# Define variables
CC=/opt/aarch64/bin/aarch64-linux-gnu-gcc
CXX=/opt/aarch64/bin/aarch64-linux-gnu-g++
ARMNN_DIR=/home/bea1e/GPU-TEE/Benchmarks/armnn
CFLAGS=-O3 -std=c++14
LDLIBS=-L$(ARMNN_DIR)/build-tool/docker_output/aarch64_build  -larmnn -larmnnOnnxParser -larmnnUtils -lprotobuf -lpthread
INCLUDES=-I$(ARMNN_DIR)/include
SRC=armnn-inference-demo.cpp

# Define target
TARGET = armnn-inference-demo

# Rule for building the target
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(LDLIBS) $(INCLUDES) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET)

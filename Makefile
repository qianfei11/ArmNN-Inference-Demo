include make.config

# Define variables
CFLAGS=-O3 -std=c++14
LDLIBS=-L$(ARMNN_DIR)/build-tool/docker_output/aarch64_build -larmnn -larmnnOnnxParser -larmnnUtils -lprotobuf -lpthread
INCLUDES=-I$(ARMNN_DIR)/include -I$(ARMNN_DIR)/profiling -I$(ARMNN_DIR)/third-party
SRC=ArmnnInferenceDemo.cpp

# Define target
TARGET = ArmnnInferenceDemo

# Rule for building the target
$(TARGET): $(SRC)
	$(CXX) $(CFLAGS) $(LDLIBS) $(INCLUDES) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET)

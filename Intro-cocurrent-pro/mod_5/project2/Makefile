IDIR=./includes
EXECUTABLES_DIR=./bin
SOURCE_CODE_DIR=./includes
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include --std c++17

.PHONY: clean build run

build: $(SOURCE_CODE_DIR)/simple.cu
	mkdir -p $(EXECUTABLES_DIR)
	$(COMPILER) $(COMPILER_FLAGS) $(SOURCE_CODE_DIR)/simple.cu -o $(EXECUTABLES_DIR)/complex.exe

clean:
	rm -rf $(EXECUTABLES_DIR)
	rm -f complex.exe

run:
	$(EXECUTABLES_DIR)/complex.exe

all: clean build run
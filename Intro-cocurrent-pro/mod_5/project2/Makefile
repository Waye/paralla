IDIR=./includes
EXECUTABLES_DIR=./bin
SOURCE_CODE_DIR=./src
COMPILER=REPLACE_WITH_NVIDIA_CROSS_COMPILER
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: $(SOURCE_CODE_DIR)/simple.cu
	mkdir $(EXECUTABLES_DIR)
	$(COMPILER) $(COMPILER_FLAGS) $(SOURCE_CODE_DIR)/simple.cu -o $(EXECUTABLES_DIR)/simple.exe

clean:
	rm -f simple.exe

run:
	$(EXECUTABLES_DIR)/simple.exe

all: clean build run

# Compiler
CC := g++-6

CFLAGS := -Wall -pedantic
ROOT_DIR := $(shell pwd)
SRC := $(ROOT_DIR)/src
INCLUDE := $(ROOT_DIR)/include

# If you want to compile for CUDA (GPU) set to 1
USE_CUDA := 0

ifeq ($(USE_CUDA), 1)
	$(CFLAGS) += -DGPU=1
endif

all:
    $(CC)

# Test
.PHONY: test
test:
	@echo "Needs to be implemented"

# Clean up 
.PHONY: clean
clean:
	rm *a.out

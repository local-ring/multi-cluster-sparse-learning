#!/bin/bash

PYTHON_BIN=$(which python)
PYTHON_CONFIG=$(which python3-config)

# Properly preserve multiple -I flags
PYTHON_INCLUDES=$($PYTHON_CONFIG --includes)

# NumPy include
NUMPY_INCLUDE=$($PYTHON_BIN -c "import numpy; print(numpy.get_include())")

# Path to gfortran lib from Homebrew
GFORTRAN_LIB="/opt/homebrew/opt/gcc/lib/gcc/current"

# Source files
SRC_FILES="c/main_wrapper.c c/head_tail_proj.c c/fast_pcst.c c/sort.c"
OUTPUT_PATH="../sparse_module.so"

echo "Compiling sparse_module.so using Python: $PYTHON_BIN"

gcc -g -shared -Wall -Wextra -fPIC -std=c11 -O3 \
${PYTHON_INCLUDES} -I${NUMPY_INCLUDE} \
-L${GFORTRAN_LIB} \
-o ${OUTPUT_PATH} ${SRC_FILES} \
-lm -lpthread -lgfortran -undefined dynamic_lookup

echo "✅ Done. Output written to ${OUTPUT_PATH}"

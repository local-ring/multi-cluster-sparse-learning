cat > algo_wrapper/build.sh <<'EOF'
#!/bin/bash
set -e

# Determine script and project dirs
SCRIPT_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
PROJECT_ROOT="\$(cd "\$SCRIPT_DIR/.." && pwd)"

# Use whichever python3 is on your PATH
PYTHON=\$(which python3)
PYTHON_CONFIG=\$(which python3-config)
echo "Building with Python: \$PYTHON"

# Compile flags
CC=gcc
CFLAGS="-Wall -Wextra -O3 -fPIC -std=c11"
PY_INCLUDES="\$($PYTHON_CONFIG --includes)"
PY_LDFLAGS="\$($PYTHON_CONFIG --ldflags --embed)"
NUMPY_INC="\$($PYTHON -c 'import numpy; print(numpy.get_include())')"

# Source files
SRC_DIR="\$SCRIPT_DIR/c"
SRC="\$SRC_DIR/main_wrapper.c \$SRC_DIR/head_tail_proj.c \$SRC_DIR/fast_pcst.c \$SRC_DIR/sort.c"

# Always write into the actual project root
OUTPUT="\$PROJECT_ROOT/sparse_module.so"

echo "Compiling -> \$OUTPUT"
\$CC \$CFLAGS \$PY_INCLUDES -I\$NUMPY_INC -shared \
    \$SRC \$PY_LDFLAGS -lm

echo "✅ Built -> sparse_module.so"
EOF

chmod +x algo_wrapper/build.sh

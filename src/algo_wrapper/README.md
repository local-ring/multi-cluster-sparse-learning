# Building sparse_module.so

This repository includes a C extension used for sparse structured learning, compiled as `sparse_module.so`. The build process is managed via `build.sh`.

## For Linux Users
This build process only works for macOS (Apple Silicon). For Linux systems, please refer to the original implementation by the author at: https://github.com/baojian/dmo-fw.


## Requirements

Activate the `gfl` Conda environment and install the following:

```bash
conda activate gfl
```
Then install the necessary system-level dependencies:
```bash
brew install gcc
```
Ensure `libgfortran` is available at `/opt/homebrew/opt/gcc/lib/gcc/<version>`.

## Compilation

Run the build script from the `algo_wrapper/` directory:

```bash
chmod +x build.sh
./build.sh
```

This compiles the sources under `c/` and produces `sparse_module.so` in `algo_wrapper/`.

## Validation

To check the build:

```bash
otool -L sparse_module.so
```

You should see something like:

```
../sparse_module.so:
    /opt/homebrew/opt/gcc/lib/gcc/current/libgfortran.5.dylib
    /usr/lib/libSystem.B.dylib
```
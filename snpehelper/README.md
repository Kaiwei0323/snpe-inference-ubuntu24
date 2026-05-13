# Build libsnpehelper

The resulting `libsnpehelper.so` must load the same **host** `libSNPE.so` major version as the **Hexagon skels** under `/usr/lib/rfsa/adsp` (from the `libsnpe1` package on QCS6490 Ubuntu). If you link against an older SDK under `/data/sdk/...`, SNPE may report `setTargetRuntime: Selected runtime not present` and fall back to CPU.

CMake prefers `/usr/lib/libSNPE.so` when that file exists; otherwise it uses `SNPE_ROOT` in `CMakeLists.txt`.

## Build

From this directory:

```bash
rm -rf build && mkdir build && cd build
cmake ..
cmake --build . -j"$(nproc)"
```

This produces `build/libsnpehelper.so`.

## Install into the Flask tutorial

```bash
cp -f build/libsnpehelper.so ../Tutorials/snpe/libsnpehelper.so
```

On-device, `Tutorials/setup.sh` runs this rebuild automatically after `libsnpe1` is installed.

## Dependencies

- `cmake`, `build-essential`, `python3-dev`, `python3-pybind11`
- SNPE C++ headers under `SNPE_ROOT/include/SNPE` (default: tree under `/data/sdk/...` in `CMakeLists.txt`)

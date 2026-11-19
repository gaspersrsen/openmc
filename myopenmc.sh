#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$script_dir/build"

cmake -S "$script_dir" -B "$build_dir" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DOPENMC_USE_MPI=ON

cmake --build "$build_dir" --parallel 32
sudo cmake --install "$build_dir"

echo "Successfully installed OpenMC"
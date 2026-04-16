#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir build
cd build

cmake -DHDF5_PREFER_PARALLEL=on -DOPENMC_USE_MPI=on ..
make -j32
sudo make install -j32
echo "Successfully installed OpenMC"
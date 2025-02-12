#!/bin/bash
# This script installs OpenMC with Embree support.
# It is used in the devcontainer to install OpenMC and all its dependencies to the standard location ($INSTALL_DIR).
# If needed, script can be used to produce wheels for OpenMC and PyMOAB.
# In that case, run as
#  $ .devcontainer/install_openmc_with_embree.sh --build-wheels=true
# In that case the wheels will be saved in the $HOME/wheels directory.
# set -euo pipefail

# Default parameter values
BUILD_WHEELS=false
# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build-wheels=*)
            BUILD_WHEELS="${1#*=}" # Extract value after '='
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

if [[ "$BUILD_WHEELS" != "true" && "$BUILD_WHEELS" != "false" ]]; then
    echo "Invalid value for --build-wheels. Allowed values are 'true' or 'false'."
    exit 1
fi

# Define the source directory in the home folder
SRC_DIR="$HOME/src"
# Install OpenMC (C++ core)
cd $SRC_DIR
if build_exists "$SRC_DIR/openmc/build"; then
    echo "OpenMC already built."
else
    PINNED_COMMIT="de8132a5a431660f5ff515cc7894ea0f283d3bec"
    #git clone --recurse-submodules --single-branch --branch develop --depth 1 https://github.com/openmc-dev/openmc.git || echo "OpenMC already cloned."
    #git clone --recurse-submodules --single-branch --branch $openmc_branch --depth 1 $OPENMC_REPO || echo "OpenMC already cloned."
    cd openmc
    #git fetch --depth 1 origin $PINNED_COMMIT
    #git checkout $PINNED_COMMIT
    git submodule update --init --recursive
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local/ \
        -DCMAKE_BUILD_TYPE=Release \
        -DOPENMC_USE_DAGMC=ON \
        -DOPENMC_USE_MPI=ON \
        -DHDF5_PREFER_PARALLEL=ON \
        -DCPP20=ON \
        -DBUILD_TESTING=OFF \
        -DOPENMC_USE_LIBMESH=$build_libmesh \
        -DCMAKE_PREFIX_PATH=/usr/local/ \
        -DXTENSOR_USE_TBB=OFF \
        -DXTENSOR_USE_OPENMP=ON \
        -DXTENSOR_USE_XSIMD=OFF
    # -DCMAKE_PREFIX_PATH="/usr/local;${LIBMESH_INSTALL_DIR}" \
    # Continue installation even if the build failed. At the moment, the build fails on 90% because
    # it can not find catch2 lib when building the tests.
    make 2>/dev/null -j${compile_cores} install
    #cmake --build . --parallel "$(nproc)"
    #|| echo "Build failed, continuing to installation."
    #cmake --install .
fi

if [ "$BUILD_WHEELS" = true ]; then
    # Build and install OpenMC wheel
    cd $SRC_DIR/openmc
    python -m build --wheel --outdir "$WHEEL_DIR"

    ls "$WHEEL_DIR"
fi

#deactivate
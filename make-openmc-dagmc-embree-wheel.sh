#!/bin/bash
# This script installs OpenMC with Embree support.
# It is used in the devcontainer to install OpenMC and all its dependencies to the standard location ($INSTALL_DIR).
# If needed, script can be used to produce wheels for OpenMC and PyMOAB.
# In that case, run as
#  $ .devcontainer/install_openmc_with_embree.sh --build-wheels=true
# In that case the wheels will be saved in the $HOME/wheels directory.
set -euo pipefail

# Default parameter values
BUILD_WHEELS=true
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
mkdir -p $SRC_DIR
if [ "$BUILD_WHEELS" = true ]; then
    WHEEL_DIR="$HOME/wheels"
    rm -rf $WHEEL_DIR
    mkdir -p $WHEEL_DIR
fi

# Install system dependencies
apt-get install -y cmake \
                        g++ \
                        gfortran \
                        git \
                        hdf5-tools \
                        imagemagick \
                        libeigen3-dev \
                        libgles2-mesa-dev \
                        libglfw3 \
                        libglfw3-dev \
                        libhdf5-mpich-dev \
                        libhdf5-serial-dev \
                        libmpich-dev \
                        libmetis-dev \
                        libnetcdf-dev \
                        libnetcdf-mpi-dev \
                        libopenblas-dev \
                        libpng-dev \
                        libtbb-dev \
                        mpich \
                        wget \
                        doxygen \
                        debhelper-compat=13 \
                        mpi-default-bin \
                        libc6-dev \
                        apt-utils


#apt-get update -y
# Install ParMETIS
cd $SRC_DIR

#wget http://ftp.debian.org/debian/pool/non-free/p/parmetis/libparmetis4.0_4.0.3-7+b2_amd64.deb
#echo($ls)
#apt install ./libparmetis4.0_4.0.3-7+b2_amd64.deb
# wget http://ftp.debian.org/debian/pool/non-free/p/parmetis/libparmetis-dev_4.0.3-7+b2_amd64.deb
# dpkg -i libparmetis-dev_4.0.3-7+b2_amd64.deb
mkdir parmetis && cd parmetis
wget http://deb.debian.org/debian/pool/non-free/p/parmetis/parmetis_4.0.3.orig.tar.gz
#tar -xvzf parmetis_4.0.3.orig.tar.gz
#cd parmetis-4.0.3/
wget http://deb.debian.org/debian/pool/non-free/p/parmetis/parmetis_4.0.3-7.debian.tar.xz
#tar -xvf parmetis_4.0.3-7.debian.tar.xz
wget http://ftp.debian.org/debian/pool/non-free/p/parmetis/parmetis_4.0.3-7.dsc
dpkg-source -x parmetis_4.0.3-7.dsc
cd parmetis-4.0.3
dpkg-buildpackage -rfakeroot -b
dpkg -i ../libparmetis4.0_4.0.3-7_amd64.deb
dpkg -i ../libparmetis-dev_4.0.3-7_amd64.deb

# make config shared=1 -j"$(nproc)"
# make install
# cd metis
# make config prefix=$SRC_DIR/metis
# make install
# pip3 install metis
# export METIS_DLL=/usr/local/lib/libparmetis.so >> ~/.bashrc
# export METIS_IDXTYPEWIDTH=64  >> ~/.bashrc
# export METIS_REALTYPEWIDTH=64  >> ~/.bashrc

# Function to check if a build already exists
function build_exists {
    local build_dir="$1"
    if [ -d "$build_dir" ]; then
        echo "Build directory $build_dir exists. Skipping..."
        return 0
    else
        return 1
    fi
}


# Create and activate Python virtual environment
# VENV_DIR="venv_openmc"
# python3 -m venv $VENV_DIR
# source $VENV_DIR/bin/activate

python3 -m ensurepip --upgrade
# Upgrade pip and setuptools inside the virtual environment
pip install --upgrade pip setuptools wheel build
pip install packaging Cython
pip install numpy

# Install Embree
cd $SRC_DIR
if build_exists "$SRC_DIR/embree/build"; then
    echo "Embree already built."
else
    git clone --shallow-submodules --single-branch --branch v3.12.2 --depth 1 https://github.com/embree/embree.git || echo "Embree already cloned."
    mkdir -p embree/build
    cd embree/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ \
             -DCMAKE_BUILD_TYPE=Release \
             -DEMBREE_ISPC_SUPPORT=OFF \
             -DEMBREE_TUTORIALS=OFF \
             -DEMBREE_TUTORIALS_GLFW=OFF
    cmake --build . --parallel "$(nproc)"
    cmake --install .
fi

# Install MOAB (with PyMOAB enabled)
cd $SRC_DIR
if build_exists "$SRC_DIR/moab/build"; then
    echo "MOAB already built."
else
    #git clone --single-branch -b 5.5.1 --depth 1 https://bitbucket.org/fathomteam/moab/ || echo "MOAB already cloned."
    git clone --single-branch -b master --depth 1 https://bitbucket.org/fathomteam/moab/ || echo "MOAB already cloned."
    mkdir -p moab/build
    cd moab/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ \
             -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_HDF5=ON \
             -DENABLE_PYMOAB=ON \
             -DENABLE_BLASLAPACK=OFF \
             -DENABLE_FORTRAN=OFF \
             -DENABLE_METIS=ON \
             -DENABLE_MPI=ON \
             -DENABLE_NETCDF=ON \
             -DENABLE_PARMETIS=ON \
             -DENABLE_PNETCDF=OFF
    cmake --build . --parallel "$(nproc)"
    cmake --install .
fi

if [ "$BUILD_WHEELS" = true ]; then
    # Build and install PyMOAB wheel

    chmod -R 777 $SRC_DIR/moab/build
    cd $SRC_DIR/moab/build/pymoab
    python -m build --wheel --outdir "$WHEEL_DIR"
fi

# Install Double-Down
cd $SRC_DIR
if build_exists "$SRC_DIR/double-down/build"; then
    echo "Double-Down already built."
else
    git clone --shallow-submodules --single-branch --branch v1.1.0 --depth 1 https://github.com/pshriwise/double-down.git || echo "Double-Down already cloned."
    mkdir -p double-down/build
    cd double-down/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ \
             -DCMAKE_BUILD_TYPE=Release \
             -DMOAB_DIR=/usr/local/ \
             -DEMBREE_DIR=/usr/local/
    cmake --build . --parallel "$(nproc)"
    cmake --install .
fi

# Install DAGMC
cd $SRC_DIR
if build_exists "$SRC_DIR/DAGMC/build"; then
    echo "DAGMC already built."
else
    #git clone --single-branch --branch v3.2.4 --depth 1 https://github.com/svalinn/DAGMC.git || echo "DAGMC already cloned."
    git clone --single-branch --branch develop --depth 1 https://github.com/svalinn/DAGMC.git || echo "DAGMC already cloned."
    mkdir -p DAGMC/build
    cd DAGMC/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/ \
             -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TALLY=ON \
             -DBUILD_TESTS=OFF \
             -DBUILD_EXE=OFF \
             -DBUILD_BUILD_OBB=OFF \
             -DMOAB_DIR=/usr/local/ \
             -DDOUBLE_DOWN=ON \
             -DDOUBLE_DOWN_DIR=/usr/local/ \
             -DOpenMP_pthread_LIBRARY=/lib/x86_64-linux-gnu/libpthread.so.0 \
             -DBUILD_STATIC_EXE=OFF \
             -DBUILD_STATIC_LIBS=OFF
    cmake --build . --parallel "$(nproc)"
    cmake --install .
fi

# Install LIBMESH
# apt-get -y install m4 libnetcdf-dev libpnetcdf-dev 
# cd $SRC_DIR
# if build_exists "$SRC_DIR/libmesh/build"; then
#     echo "LIBMESH already built."
# else
#     git clone --shallow-submodules --recurse-submodules --single-branch -b "v1.8.0-rc2" --depth 1 "https://github.com/libMesh/libmesh"
#     mkdir build && cd build
#     ../libmesh/configure \
#                 --prefix=$/usr/local/ CXX=mpicxx CC=mpicc FC=mpifort F77=mpif77 \
#                 --enable-exodus \
#                 --enable-mpi \
#                 --enable-silent-rules \
#                 --enable-unique-id \
#                 --disable-eigen \
#                 --disable-fortran \
#                 --disable-lapack \
#                 --disable-examples \
#                 --disable-warnings \
#                 --disable-maintainer-mode \
#                 --disable-metaphysicl \
#                 --with-methods="opt" \
#                 --without-gdb-command \
#                 --with-cxx-std-min=2014
#     make 2>/dev/null -j${compile_cores} install
# fi


# # Install OpenMC (C++ core)
# cd $SRC_DIR
# if build_exists "$SRC_DIR/openmc/build"; then
#     echo "OpenMC already built."
# else
#     PINNED_COMMIT="de8132a5a431660f5ff515cc7894ea0f283d3bec"
#     #git clone --recurse-submodules --single-branch --branch develop --depth 1 https://github.com/openmc-dev/openmc.git || echo "OpenMC already cloned."
#     #git clone --recurse-submodules --single-branch --branch $openmc_branch --depth 1 $OPENMC_REPO || echo "OpenMC already cloned."
#     cd openmc
#     #git fetch --depth 1 origin $PINNED_COMMIT
#     #git checkout $PINNED_COMMIT
#     git submodule update --init --recursive
#     mkdir -p build
#     cd build
#     cmake .. \
#         -DCMAKE_INSTALL_PREFIX=/usr/local/ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DOPENMC_USE_DAGMC=ON \
#         -DDAGMC_ROOT=/usr/local/ \
#         -DOPENMC_USE_MPI=ON \
#         -DHDF5_PREFER_PARALLEL=ON \
#         -DCPP20=ON \
#         -DBUILD_TESTING=OFF \
#         -DOPENMC_USE_LIBMESH=$build_libmesh \
#         -DCMAKE_PREFIX_PATH=/usr/local \
#         -DXTENSOR_USE_TBB=OFF \
#         -DXTENSOR_USE_OPENMP=ON \
#         -DXTENSOR_USE_XSIMD=OFF
#     # -DCMAKE_PREFIX_PATH="/usr/local;${LIBMESH_INSTALL_DIR}" \
#     # Continue installation even if the build failed. At the moment, the build fails on 90% because
#     # it can not find catch2 lib when building the tests.
#     make 2>/dev/null -j${compile_cores} install
#     make install
#     #cmake --build . --parallel "$(nproc)"
#     #|| echo "Build failed, continuing to installation."
#     #cmake --install .
# fi

# # if [ "$BUILD_WHEELS" = true ]; then
# #     # Build and install OpenMC wheel
# #     cd $SRC_DIR/openmc
# #     python -m build --wheel --outdir "$WHEEL_DIR"

# #     ls "$WHEEL_DIR"
# # fi

# #deactivate
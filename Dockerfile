# To build with OpenMC and by default this Dockerfile builds the master branch of OpenMC.
# docker build -t openmc .

# To build with OpenMC develop branch
# docker build -t openmc_develop --build-arg openmc_branch=develop .

# To build with OpenMC and DAGMC enabled
# docker build -t openmc_dagmc --build-arg build_dagmc=on --build-arg compile_cores=4 .

# To build with OpenMC and Libmesh enabled
# docker build -t openmc_libmesh --build-arg build_libmesh=on --build-arg compile_cores=4 .

# To build with both DAGMC and Libmesh enabled
# docker build -t openmc_dagmc_libmesh --build-arg build_dagmc=on --build-arg build_libmesh=on --build-arg compile_cores=4 .

# sudo docker run image_name:tag_name or ID with no tag sudo docker run ID number


# global ARG as these ARGS are used in multiple stages
# By default eight cores is used to compile
ARG compile_cores=8

# By default this Dockerfile builds OpenMC without DAGMC and LIBMESH support
ARG build_dagmc=off
ARG build_libmesh=off

FROM debian:bookworm-slim AS dependencies

ARG compile_cores
ARG build_dagmc
ARG build_libmesh

# Set default value of HOME to /root
ENV HOME=/root

# ENV EMBREE_INSTALL_DIR=$HOME
# ENV DD_INSTALL_DIR=$HOME/src
######ENV DAGMC_INSTALL_DIR=$HOME/src
#ENV DAGMC_INSTALL_DIR=$HOME/src/DAGMC

# LIBMESH variables
# ENV LIBMESH_TAG="v1.7.1"
# #'v1.8.0'
# ENV LIBMESH_REPO='https://github.com/libMesh/libmesh'
# ENV LIBMESH_INSTALL_DIR=$HOME/src/LIBMESH

# NJOY variables
#ENV NJOY_REPO='https://github.com/njoy/NJOY2016'

# Setup environment variables for Docker image
# ENV LD_LIBRARY_PATH=${DAGMC_INSTALL_DIR}/lib:$LD_LIBRARY_PATH \
#     OPENMC_ENDF_DATA=/root/endf-b-vii.1 \
ENV DEBIAN_FRONTEND=noninteractive

# Install and update dependencies from Debian package manager
RUN apt-get update -y && \
    apt-get install -y \
        python3-pip python-is-python3 wget git build-essential cmake \
        mpich libmpich-dev libhdf5-serial-dev libhdf5-mpich-dev \
        libpng-dev python3-venv && \
    apt-get autoremove

# create virtual enviroment to avoid externally managed environment error
RUN python3 -m venv openmc_venv
ENV PATH=/openmc_venv/bin:$PATH

# Update system-provided pip
RUN pip install --upgrade pip
RUN pip install vtk

# Clone and install NJOY2016
RUN cd $HOME \
    && git clone --single-branch --depth 1 'https://github.com/njoy/NJOY2016' \
    && cd NJOY2016 \
    && mkdir build \
    && cd build \
    && cmake -Dstatic=on .. \
    && make 2>/dev/null -j${compile_cores} install \
    && rm -rf $HOME/NJOY2016


# RUN if [ "$build_libmesh" = "on" ]; then \
#         # Install addition packages required for LIBMESH
#         apt-get -y install m4 libnetcdf-dev libpnetcdf-dev \
#         # Install LIBMESH
#         && mkdir -p ${LIBMESH_INSTALL_DIR} && cd ${LIBMESH_INSTALL_DIR} \
#         && git clone --shallow-submodules --recurse-submodules --single-branch -b ${LIBMESH_TAG} --depth 1 ${LIBMESH_REPO} \
#         && mkdir build && cd build \
#         && ../libmesh/configure \
#                     --prefix=${LIBMESH_INSTALL_DIR} CXX=mpicxx CC=mpicc FC=mpifort F77=mpif77 \
#                     --enable-exodus \
#                     --enable-mpi \
#                     --enable-silent-rules \
#                     --enable-unique-id \
#                     --disable-eigen \
#                     --disable-fortran \
#                     --disable-lapack \
#                     --disable-examples \
#                     --disable-warnings \
#                     --disable-maintainer-mode \
#                     --disable-metaphysicl \
#                     --with-methods="opt" \
#                     --without-gdb-command \
#                     --with-cxx-std-min=2014 \
#         && make 2>/dev/null -j${compile_cores} install \
#         && rm -rf ${LIBMESH_INSTALL_DIR}/build ${LIBMESH_INSTALL_DIR}/libmesh ; \
#     fi

#FROM dependencies AS build

ARG openmc_branch=th
ENV OPENMC_REPO='https://github.com/gaspersrsen/openmc.git'

ENV HOME=/root
ARG CACHEBUST=1
#RUN echo "$CACHEBUST"

RUN mkdir -p ${HOME}/OpenMC && cd ${HOME}/OpenMC \
    && git clone --shallow-submodules --recurse-submodules --single-branch -b ${openmc_branch} --depth=1 ${OPENMC_REPO} \
    && mkdir build && cd build ; \
    if [ ${build_dagmc} = "off" ] && [ ${build_libmesh} = "off" ]; then \
        cmake ../openmc \
            -DCMAKE_CXX_COMPILER=mpicxx \
            -DOPENMC_USE_MPI=on \
            -DHDF5_PREFER_PARALLEL=on ; \
    fi ; \
    make 2>/dev/null -j${compile_cores} install \
    && cd ../openmc && pip install .[test,depletion-mpi] \
    && python -c "import openmc"


# clone and install openmc
# RUN mkdir -p ${HOME}/src && cd ${HOME}/src \
#     && git clone --shallow-submodules --recurse-submodules --single-branch -b ${openmc_branch} --depth=1 ${OPENMC_REPO} \
#     && cd openmc \
#     && chmod u+r+x make-openmc-dagmc-embree-wheel.sh ; \
#     #&& echo "export DAGMC_DIR=$HOME/DAGMC" >> ~/.bashrc \
#     # if [ "$build_dagmc" = "on" ]; then \
#     #     ./make-openmc-dagmc-embree-wheel.sh ; \
#     # fi

# RUN cd ${HOME}/src/openmc \
#     && chmod u+r+x openmc_installer.sh ; \
#     if [ "$build_dagmc" = "on" ]; then \
#         ./openmc_installer.sh ; \
#     fi ; \
#     pip install .[test,depletion-mpi] \
#     && python -c "import openmc"
#make 2>/dev/null -j${compile_cores} install
#echo "Hi"
# git submodule update --init --recursive \
# && mkdir -p build \
# && cd build \
# && cmake .. \
#     -DCMAKE_INSTALL_PREFIX=/usr/local/ \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_COMPILER=mpicxx \
#     -DOPENMC_USE_MPI=ON \
#     -DHDF5_PREFER_PARALLEL=ON \
#     -DOPENMC_USE_DAGMC=$build_dagmc \
#     -DOPENMC_USE_LIBMESH=$build_libmesh \
#     -DCPP20=ON \
#     -DBUILD_TESTING=OFF \
#     -DCMAKE_PREFIX_PATH="${DAGMC_INSTALL_DIR};${LIBMESH_INSTALL_DIR}" \
#     -DXTENSOR_USE_TBB=OFF \
#     -DXTENSOR_USE_OPENMP=ON \
#     -DXTENSOR_USE_XSIMD=OFF; \
#make 2>/dev/null -j${compile_cores} install

# FROM dependencies AS build
# ENV HOME=/root
# ARG CACHEBUST=1
# RUN echo "$CACHEBUST"
    
    # Continue installation even if the build failed. At the moment, the build fails on 90% because
    # it can not find catch2 lib when building the tests.
    # cmake --build . --parallel ${compile_cores} || echo "Build failed, continuing to installation." \
    # && cmake --install .
#     fi ; \
#     mkdir build && cd build ; \
#     if [ ${build_dagmc} = "on" ] && [ ${build_libmesh} = "on" ]; then \
#         cmake ../openmc \
#             -DCMAKE_CXX_COMPILER=mpicxx \
#             -DOPENMC_USE_MPI=on \
#             -DHDF5_PREFER_PARALLEL=on \
#             -DOPENMC_USE_DAGMC=on \
#             -DOPENMC_USE_LIBMESH=on \
#             -DCMAKE_PREFIX_PATH="${DAGMC_INSTALL_DIR};${LIBMESH_INSTALL_DIR}" ; \
#     fi ; \
#     if [ ${build_dagmc} = "on" ] && [ ${build_libmesh} = "off" ]; then \
#         cmake ../openmc \
#             -DCMAKE_CXX_COMPILER=mpicxx \
#             -DOPENMC_USE_MPI=on \
#             -DHDF5_PREFER_PARALLEL=on \
#             -DOPENMC_USE_DAGMC=ON \
#             -DCMAKE_PREFIX_PATH=${DAGMC_INSTALL_DIR} ; \
#     fi ; \
#     if [ ${build_dagmc} = "off" ] && [ ${build_libmesh} = "on" ]; then \
#         cmake ../openmc \
#             -DCMAKE_CXX_COMPILER=mpicxx \
#             -DOPENMC_USE_MPI=on \
#             -DHDF5_PREFER_PARALLEL=on \
#             -DOPENMC_USE_LIBMESH=on \
#             -DCMAKE_PREFIX_PATH=${LIBMESH_INSTALL_DIR} ; \
#     fi ; \
#     if [ ${build_dagmc} = "off" ] && [ ${build_libmesh} = "off" ]; then \
#         cmake ../openmc \
#             -DCMAKE_CXX_COMPILER=mpicxx \
#             -DOPENMC_USE_MPI=on \
#             -DHDF5_PREFER_PARALLEL=on ; \
#     fi ; \
#     make 2>/dev/null -j${compile_cores} install \
    # && cd ../openmc && pip install .[test,depletion-mpi] \
    # && python -c "import openmc"

# # FROM build AS release

# ENV HOME=/root
# ENV OPENMC_CROSS_SECTIONS=/root/nndc_hdf5/cross_sections.xml

# # Download cross sections (NNDC and WMP) and ENDF data needed by test suite
# # RUN ${HOME}/OpenMC/openmc/tools/ci/download-xs.sh
# RUN /bin/bash -c 'echo "CMAKE_BUILD_PARALLEL_LEVEL=${compile_cores}" >> ~/.bashrc'
# RUN printf '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
# RUN /bin/bash -c 'cd $HOME \
#     && apt install flex -y \
#     && apt install bison -y \
#     && git clone https://github.com/neams-th-coe/cardinal.git \
#     && cd cardinal \
#     && echo "export ENABLE_DAGMC=yes" >> ~/.bashrc \
#     && echo "export NEKRS_HOME=$HOME/cardinal/install" >> ~/.bashrc \
#     && echo "export NEKRS_OCCA_MODE_DEFAULT=CPU" >> ~/.bashrc \
#     && echo "export CC=mpicc" >> ~/.bashrc \
#     && echo "export CXX=mpicxx" >> ~/.bashrc \
#     && echo "export FC=mpif90" >> ~/.bashrc \
#     && echo "export MPICH_FC=gfortran" >> ~/.bashrc \
#     && echo "export JOBS={compile_cores}" >> ~/.bashrc \
#     && echo "export LIBMESH_JOB={compile_cores}" >> ~/.bashrc \
#     && echo "export MOOSE_JOBS={compile_cores}" >> ~/.bashrc \
#     && ./scripts/get-dependencies.sh \
#     && ./contrib/moose/scripts/update_and_rebuild_petsc.sh \
#     && ./contrib/moose/scripts/update_and_rebuild_libmesh.sh \
#     && ./contrib/moose/scripts/update_and_rebuild_wasp.sh '

# RUN /bin/bash -c 'pip install pyyaml jinja2 packaging \
#     && cd $HOME \
#     && cd cardinal \
#     && export ENABLE_DAGMC=yes \
#     && export NEKRS_HOME=$HOME/cardinal/install \
#     && export NEKRS_OCCA_MODE_DEFAULT=OPENMP'

# ENV NEKRS_HOME=/root/cardinal/install
# RUN /bin/bash -c 'cd $HOME \
#     && cd cardinal \
#     && apt install pkg-config -y \
#     && make -j${compile_cores} MAKEFLAGS=-j${compile_cores} '

# RUN /bin/bash -c 'echo "export RUNLEVEL=1" >> ~/.bashrc \
#     && apt install libxt-dev xorg -y \
#     && cd $HOME \
#     && git clone https://github.com/Nek5000/Nek5000.git \
#     && cd Nek5000/tools \
#     && ./maketools all \
#     && echo "export PATH=/etc:$PATH" >> ~/.bashrc \
#     && echo "export PATH=/root/Nek5000/bin:/etc:$PATH" >> ~/.bashrc \
#     && echo "export PATH=/root/cardinal/build/openmc/bin:$PATH" >> ~/.bashrc \
#     && echo "export PATH=$NEKRS_HOME/bin:$PATH" >> ~/.bashrc \
#     && echo "export PATH=/root/cardinal:$PATH" >> ~/.bashrc'
# RUN /bin/bash -c 'python -m pip install git+https://github.com/openmc-dev/openmc_cad_adapter.git'
# ENV OPENMC_CROSS_SECTIONS=$HOME/nndc_hdf5/cross_sections.xml
# ENV NEKRS_HOME=$HOME/cardinal/install
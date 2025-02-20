# THIS DOCKER-FILE CREATES DOCKER CONTAINER FOR MULTIPHYISCS SIMULATIONS
# USING CARDINAL AND MOOSE SOFTWARES

# global ARG as these ARGS are used in multiple stages
# By default one core is used to compile
ARG compile_cores=1

FROM debian:bookworm-slim AS dependencies

ARG compile_cores
ARG build_dagmc
ARG build_libmesh

# Set default value of HOME to /root
ENV HOME=/root
ENV DEBIAN_FRONTEND=noninteractive

# Install and update dependencies from Debian package manager
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y && \
    apt-get upgrade -y && \
    apt-get install -y \
        python3-pip python-is-python3 wget git build-essential cmake \
        mpich libmpich-dev libhdf5-serial-dev libhdf5-mpich-dev \
        libpng-dev python3-venv pkg-config && \
    apt-get autoremove

# create virtual enviroment to avoid externally managed environment error
RUN python3 -m venv openmc_venv
ENV PATH=/openmc_venv/bin:$PATH

# Update system-provided pip
RUN pip install --upgrade pip
RUN pip install pyyaml jinja2 packaging

# Clone and install NJOY2016
RUN cd $HOME \
    && git clone --single-branch --depth 1 'https://github.com/njoy/NJOY2016' \
    && cd NJOY2016 \
    && mkdir build \
    && cd build \
    && cmake -Dstatic=on .. \
    && make 2>/dev/null -j${compile_cores} install \
    && rm -rf $HOME/NJOY2016

#FROM dependencies AS build

ARG openmc_branch=th
ENV OPENMC_REPO='https://github.com/gaspersrsen/openmc.git'
ENV CARDINAL_REPO='https://github.com/gaspersrsen/cardinal.git'

ENV HOME=/root
ARG CACHEBUST=1

ENV JOBS=${compile_cores}
ENV MOOSE_JOBS=${compile_cores}
ENV LIBMESH_JOBS=${compile_cores}
ENV METHODS=opt
ENV ENABLE_DAGMC=yes
ENV NEKRS_OCCA_MODE_DEFAULT=CPU
ENV NEKRS_HOME=$HOME/cardinal/install
ENV CC=mpicc
ENV CXX=mpicxx
ENV FC=mpif90
#RUN echo "$CACHEBUST"

RUN mkdir -p $HOME \
    && cd $HOME \
    && export CC=mpicc CXX=mpicxx FC=mpif90 F90=mpif90 F77=mpif77 \
        NEKRS_HOME=$HOME/cardinal/install \
        NEKRS_OCCA_MODE_DEFAULT=CPU \
        JOBS=${compile_cores} \
        MOOSE_JOBS=${compile_cores} \
        LIBMESH_JOBS=${compile_cores} \
        METHODS=opt \
        ENABLE_DAGMC=yes \
    && git clone -b master $CARDINAL_REPO \
    && apt-get install -y \
        flex bison \
    && cd ./cardinal \
    && git submodule foreach git pull \
    && ./scripts/get-dependencies.sh \
    && cd ./contrib/moose/ \
    && git checkout master \
    && git pull origin \
    && cd $HOME/cardinal \
    && ./contrib/moose/scripts/update_and_rebuild_petsc.sh \
    && ./contrib/moose/scripts/update_and_rebuild_libmesh.sh \
    && ./contrib/moose/scripts/update_and_rebuild_wasp.sh \
    
RUN cd $HOME/cardinal \
    && make -j${compile_cores} MAKEFLAGS=-j${compile_cores} \
    && export HDF5_ROOT=$HOME/cardinal/contrib/moose/petsc/arch-moose/externalpackages/hdf5-1.14.3-p1
    #&& export HDF5_ROOT=$HOME/cardinal/contrib/moose/petsc/arch-moose/lib/

ENV HDF5_ROOT=$HOME/cardinal/contrib/moose/petsc/arch-moose/externalpackages/hdf5-1.14.3-p1
#ENV HDF5_ROOT=$HOME/cardinal/contrib/moose/petsc/arch-moose/lib/

RUN cd $HOME/cardinal/contrib/openmc \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DOPENMC_USE_MPI=on \
        -DHDF5_PREFER_PARALLEL=on \
        -DOPENMC_USE_DAGMC=on \
        -DOPENMC_USE_LIBMESH=on \
        -DCMAKE_PREFIX_PATH="$HOME/cardinal/install/lib/cmake/dagmc;$HOME/cardinal/contrib/moose/libmesh/build" \
    && make 2>/dev/null -j${compile_cores} install \
    && cd .. \
    && MPICC=/usr/bin/mpicc python -m pip install mpi4py \
    && CC=/usr/bin/mpicc HDF5_MPI=ON HDF5_DIR=$HDF5_ROOT python -m pip install --no-binary=h5py h5py \
    && pip install .[test,depletion-mpi] \
    && python -c "import openmc"

# RUN cd $HOME/cardinal/contrib/openmc \
#     && pip install .[test,depletion-mpi] \
#     && python -c "import openmc"


#clone and install MOOSE
# RUN export CC=mpicc CXX=mpicxx FC=mpif90 F90=mpif90 F77=mpif77 \
#     && mkdir -p ${HOME}/src && cd ${HOME}/src \
#     && git clone https://github.com/idaholab/moose.git \
#     && cd moose \
#     && git checkout master \
#     && cd ./scripts \
#     && export MOOSE_JOBS=6 METHODS=opt \
#     && ./update_and_rebuild_petsc.sh   || return \
#     && ./update_and_rebuild_libmesh.sh  || return \
#     && ./update_and_rebuild_wasp.sh  || return \
#     && cd ../test \
#     && make -j${compile_cores} \
#     && ./run_tests -j${compile_cores}

# clone and install openmc
# RUN mkdir -p ${HOME}/src && cd ${HOME}/src \
#     && git clone --shallow-submodules --recurse-submodules --single-branch -b ${openmc_branch} --depth=1 ${OPENMC_REPO} \
#     && cd openmc \
#     && chmod u+r+x make-openmc-dagmc-embree-wheel.sh ; \
#     if [ "$build_dagmc" = "on" ]; then \
#         ./make-openmc-dagmc-embree-wheel.sh ; \
#     else cmake ../openmc \
#             -DCMAKE_CXX_COMPILER=mpicxx \
#             -DOPENMC_USE_MPI=on \
#             -DHDF5_PREFER_PARALLEL=on \
#         && make 2>/dev/null -j${compile_cores} install \
#         && cd ../openmc && pip install .[test,depletion-mpi] \
#         && python -c "import openmc"; \
#     fi

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
#!/bin/bash
set -x

hash=""
# guess default DEV_SIGNATURE
is_git=$(git rev-parse --is-inside-work-tree)
if [ $is_git ]; then
    hash="git-"$(git rev-parse --short HEAD)
fi

BUILDDIR=${BUILDDIR:=/tmp/nccl-exp/2.18.3/build}
NVCC_ARCH=${NVCC_ARCH:="a100,h100"}
CUDA_HOME=${CUDA_HOME:="/usr/local/cuda"}
# TODO: automatically get hg id or git hash
DEV_SIGNATURE=${DEV_SIGNATURE:="$hash"}

echo "BUILDDIR=${BUILDDIR}"
echo "NVCC_ARCH=${NVCC_ARCH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "DEV_SIGNATURE=${DEV_SIGNATURE}"

IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
arch_gencode=""
for arch in "${arch_array[@]}"
do
    case "$arch" in
       "p100")
       arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
        ;;
       "v100")
       arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
        ;;
       "a100")
       arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
       ;;
       "h100")
        arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
       ;;
    esac
done

echo "NVCC_GENCODE=$arch_gencode"

# build libnccl
export BUILDDIR=$BUILDDIR
make src.clean && make -j src.build NVCC_GENCODE="$arch_gencode" CUDA_HOME="$CUDA_HOME" DEV_SIGNATURE="$DEV_SIGNATURE"

# sanity check
pushd examples
export NCCL_HOME=$BUILDDIR
make clean && make all
NCCL_DEBUG=WARN LD_LIBRARY_PATH=$BUILDDIR/lib $BUILDDIR/examples/HelloWorld
popd

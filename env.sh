#########################################################################
# File Name: env.sh
# Author: panzhjie
#########################################################################
#!/bin/bash

# module use /public/software/modules/


# module load compiler/intel/2021.3.0 
# module load compiler/cmake/3.25.3 
# module load sghpc-mpi-gcc/25.8
# export CMAKE_PREFIX_PATH=/public/home/sghpc_sdk/Linux_x86_64/25.8/dtk/dtk-25.04.2/dcc/comgr/lib64/cmake/amd_comgr/:$CMAKE_PREFIX_PATH
# module load compiler/intel/2019.4.243
# module load compiler/cmake/3.24.1

module load compiler/cmake/3.24.1
module load compiler/devtoolset/7.3.1

module load compiler/rocm/dtk/25.04.3
module load mpi/hpcx/2.11.0/gcc-7.3.1

export HIP_PATH=/public/software/compiler/dtk/dtk-25.04.3
export ROCM_PATH=/public/software/compiler/dtk/dtk-25.04.3
export DEVICE_LIB_PATH=/public/software/compiler/dtk/dtk-25.04.3/lib/bitcode

# export CMAKE_PREFIX_PATH=/public/software/compiler/dtk/dtk-25.04.3:/public/software/compiler/dtk/dtk-25.04.3/lib64/cmake/amd_comgr:$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/public/software/compiler/dtk/dtk-25.04.3/lib64
export LIBRARY_PATH=/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/include:$CPLUS_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=/work1/baihe/users/sumeng/kmc/vendor/pkg/git.hpcer.dev/genshen/kiwi/include:$CPLUS_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/fmtlib/fmt/include:$CPLUS_INCLUDE_PATH

# export LDFLAGS="-Wl,--whole-archive /work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/lib/libyaml-cpp.a /work1/baihe/users/sumeng/kmc/vendor/pkg/git.hpcer.dev/genshen/kiwi/lib/libkiwi.a /work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/fmtlib/fmt/lib64/libfmt.a -Wl,--no-whole-archive -lstdc++"

# export LDFLAGS="-L/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/lib -Wl,-rpath,/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/lib -L/work1/baihe/users/sumeng/kmc/vendor/pkg/git.hpcer.dev/genshen/kiwi/lib -Wl,-rpath,/work1/baihe/users/sumeng/kmc/vendor/pkg/git.hpcer.dev/genshen/kiwi/lib -L/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/fmtlib/fmt/lib64 -Wl,-rpath,/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/fmtlib/fmt/lib64 -lyaml-cpp -lkiwi -lfmt"
# export LDFLAGS="-L$(pwd)/vendor/pkg/github.com/jbeder/yaml-cpp/lib -Wl,-rpath,$(pwd)/vendor/pkg/github.com/jbeder/yaml-cpp/lib -lyaml-cpp -lfmt -lkiwi"
# export LDFLAGS="-L/work1/baihe/users/sumeng/kmc/vendor/pkg/github.com/jbeder/yaml-cpp/lib"


# export YAML_LIB=$PROJECT_ROOT/vendor/pkg/github.com/jbeder/yaml-cpp/lib
# export KIWI_LIB=$PROJECT_ROOT/vendor/pkg/git.hpcer.dev/genshen/kiwi/lib
# export FMT_LIB=$(find $PROJECT_ROOT/vendor -name "libfmt.a" -exec dirname {} \; | grep "pkg" | head -n 1)



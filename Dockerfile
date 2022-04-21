FROM maturin

WORKDIR /
RUN yum install -y openssl-devel gcc gcc-c++
RUN git clone --depth=1 https://github.com/llvm/llvm-project.git 
RUN cd llvm-project \
    && mkdir build \
    && cd build \
    && cmake -DLLVM_ENABLE_PROJECTS=clang -G "Unix Makefiles" ../llvm \
    && make

ENV LIBCLANG_PATH=/llvm-project/build/lib
ENV CC=gcc
ENV CXX=g++

WORKDIR /io
ENTRYPOINT ["/usr/bin/maturin"]


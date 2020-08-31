FROM rust

RUN apt-get update \
    && apt-get install -y gfortran \
    && apt install -y libopenblas-dev

RUN mkdir -p /kriging 
WORKDIR /kriging

COPY . ./


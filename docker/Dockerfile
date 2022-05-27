FROM debian:10.9

WORKDIR /opt/build/

#ENV PROJSRC /bayesship

ENV CPATH $CPATH:/usr/local/include:/ptrjmcmc/build/install/include
ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/lib64:/usr/local/lib:/ptrjmcmc/build/install/lib
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib:/ptrjmcmc/build/install/lib

RUN apt update && apt install -y \
build-essential \
make \
cmake \
git  \
wget \
libfftw3-dev \
libgsl-dev \
libeigen3-dev \
cmake-curses-gui \
libhdf5-dev \
vim \
automake \
autoconf \
libtool \
devscripts \
python3 \
python3-pip \
swig \
doxygen \
libgtest-dev \
libarmadillo-dev \
libboost-all-dev \
valgrind

RUN pip3 install --upgrade pip

RUN pip3 install \
numpy \
matplotlib \
scipy \
pandas

WORKDIR /opt
RUN wget https://www.mlpack.org/files/mlpack-3.4.2.tar.gz
RUN tar -zxf mlpack-3.4.2.tar.gz
RUN mkdir mlpack-3.4.2/build
WORKDIR mlpack-3.4.2/build
RUN cmake ../
RUN make -j1 
RUN make install


WORKDIR /opt
RUN git clone https://github.com/nlohmann/json.git
WORKDIR json
RUN mkdir build
WORKDIR build
RUN cmake -DJSON_BuildTests=false ../
RUN make -j 4 
RUN make install


## HDF5 install ##
#RUN mkdir /usr/local/hdf5 
#WORKDIR /usr/local/hdf5
#RUN wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_0/source/hdf5-1.12.0.tar.gz
#RUN tar -zxf hdf5-1.12.0.tar.gz  
#WORKDIR /usr/local/hdf5/hdf5-1.12.0/
#RUN ./configure --prefix=/usr/local && make check && make install && make check-install

WORKDIR /opt/build

#COPY . /ptrjmcmc

#RUN ["/bin/bash"]


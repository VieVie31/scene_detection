FROM debian:jessie

RUN sudo apt-get update
RUN sudo apt-get -y install build-essential cmake git pkg-config
RUN sudo apt-get -y install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libgtk2.0-dev
RUN apt-get -y install libatlas-base-dev gfortran
RUN apt-get -y install python3-pip
RUN apt-get -y install python3.4-dev
RUN pip3 install numpy

RUN cd ~
RUN git clone https://github.com/Itseez/opencv.git
RUN cd opencv
RUN git checkout 3.1.0

RUN cd ~
RUN git clone https://github.com/Itseez/opencv_contrib.git
RUN cd opencv_contrib
RUN git checkout 3.1.0

RUN cd ~/opencv
RUN mkdir build
RUN cd build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..

RUN make -j4
RUN make install
RUN ldconfig

ADD . /src/
WORKDIR /src/

RUN python test.py


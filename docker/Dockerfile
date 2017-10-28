FROM continuumio/miniconda3

WORKDIR /root

# OpenCV deps
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y libgomp1
RUN apt-get install -y libgl1-mesa-glx

# Anaconda environment
ADD environment.yml /root
RUN conda install -y python=3.5.4
RUN conda env update -f environment.yml
RUN pip install pytest pytest-coverage

# Clone source
RUN git clone https://github.com/USGS-Astrogeology/autocnet.git
WORKDIR /root/autocnet

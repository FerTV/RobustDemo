FROM ubuntu:22.04

ENV TZ=Europe/Madrid \
    DEBIAN_FRONTEND=noninteractive

# Install python3.11
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv

# Install curl and network tools
RUN apt-get install -y curl net-tools

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.11 get-pip.py

# Install gcc and git
RUN apt-get update && apt-get install -y gcc git

COPY requirements.txt .
# Install the required packages
RUN python3.11 -m pip install -r requirements.txt
FROM ubuntu:20.04

RUN apt-get update

RUN apt-get install -y curl python3-pip git git-lfs sudo wget
RUN python3 -m pip install --upgrade pip

RUN curl -fsSL https://get.docker.com | sh

ENV USER=steven
RUN useradd -ms /bin/bash $USER
RUN usermod -aG sudo $USER
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USER-user
RUN usermod -aG docker $USER
USER $USER
WORKDIR /home/$USER


ENV PATH="${PATH}:/home/$USER/.local/bin"
RUN pip3 install --no-cache-dir --user --upgrade duckietown-shell
RUN dts --set-version daffy

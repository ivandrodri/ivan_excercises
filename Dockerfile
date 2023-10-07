FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential \
         cmake \
         git \
         curl \
         python3-opengl \
         ca-certificates \
         supervisor xinetd x11vnc xvfb tinywm openbox xdotool wmctrl x11-utils xterm \
         libgl1-mesa-glx \
         libjpeg-dev \
         libnvidia-gl-450-server \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*


RUN conda install -y ipykernel jupyter


RUN pip install -U git+https://github.com/thu-ml/tianshou.git@master --upgrade
RUN pip install pandas matplotlib opencv-python gym atari-py

#RUN apt-get update
#RUN apt-get install -y nano \
#    carla-simulator

#RUN git clone https://github.com/rail-berkeley/d4rl.git
#WORKDIR d4rl
#RUN pip install -e .
#WORKDIR ..

RUN pip install -U git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl --upgrade

RUN curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --output mujoco210.tar.gz
RUN mkdir /root/.mujoco
RUN tar -xf mujoco210.tar.gz --directory /root/.mujoco


RUN pip3 install -U Cython==0.29
RUN pip3 install -U minari==0.4.1
RUN pip3 install -U gymnasium-robotics==1.2.2


RUN apt-get update
#RUN sudo apt-get install -y libosmesa6-dev
RUN apt-get install -y libglu1-mesa-dev mesa-common-dev \
    libosmesa6-dev \
    nano



#RUN pip3 install -U carla==0.9.5

RUN chmod -R a+rx /root/.mujoco

RUN echo "export PYTHONPATH=\$PYTHONPATH:/tianshou" >> ~/.bashrc

RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc

RUN useradd -u 1000 -m -s /bin/bash tianshou_dev
WORKDIR /home/tianshou_dev
USER tianshou_dev


# Create conda user, get anaconda by web or locally
#RUN useradd --create-home --home-dir /home/condauser --shell /bin/bash condauser

# Setup our environment for running the ipython notebook
# Setting user here makes sure ipython notebook is run as user, not root
#USER condauser
#ENV HOME=/home/condauser
#ENV SHELL=/bin/bash
#ENV USER=condauser
#WORKDIR /home/condauser/notebooks

RUN mkdir ~/.vnc && touch ~/.vnc/passwd
RUN x11vnc -storepasswd "vncdocker" ~/.vnc/passwd

#EXPOSE 5900



#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
#ENV PYTHONPATH=/tianshou/:$PYTHONPATH

COPY entry.sh /tianshou/entry.sh
#RUN chmod +x /tianshou/entry.sh



CMD ["/usr/bin/x11vnc", "-forever", "-usepw", "-create"]


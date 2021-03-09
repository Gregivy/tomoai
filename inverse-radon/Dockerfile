FROM tensorflow/tensorflow:latest-gpu
RUN mkdir -p /project
WORKDIR /project
COPY ./requirements.txt ./requirements.txt
RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y
RUN apt install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN conda install --channel defaults conda python=3.6 --yes
RUN conda update --channel defaults --all --yes
RUN conda install -c conda-forge tomopy --yes
RUN pip install -r requirements.txt

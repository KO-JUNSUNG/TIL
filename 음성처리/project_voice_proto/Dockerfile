# docker run -it -v /호스트/공유폴더:/컨테이너내부경로 컨테이너이미지명
# sudo docker run -it -v /mnt/c/Users/KOJUNSUNG/Desktop/lab/torchlab/project:/app/voice db8aaa42a
# sudo docker run -it --rm -v /mnt/c/Users/KOJUNSUNG/Desktop/lab/torchlab/project:/app/ 7279387a98f7
FROM tensorflow/tensorflow:2.7.0-gpu

USER root
WORKDIR /app


# Nvidia GPG Public Key 교체 
RUN rm /etc/apt/sources.list.d/cuda.list 
#   && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
	  sudo \
	  apt-utils 

RUN apt-get install -y build-essential libsndfile-dev \
    && apt-get install -y llvm-9-dev llvm-9-runtime llvm-9-tools libedit-dev \
    && python3 -m pip install --upgrade pip
RUN pip install ipython numba==0.51.2 llvmlite==0.34.0 numpy==1.19.5\
    && apt-get install -y zsh tmux wget git libsndfile1 cmake

COPY ./TTS.py /app/
RUN git clone https://github.com/TensorSpeech/TensorFlowTTS.git 
RUN git clone https://github.com/repodiac/german_transliterate.git
RUN pip install tensorflow-addons==0.15.0
RUN cd TensorFlowTTS \
    && pip install .
RUN pip install TensorFlowTTS soundfile pyopenjtalk

RUN chmod -R 777 /app

ENTRYPOINT [ "python3","TTS.py" ]

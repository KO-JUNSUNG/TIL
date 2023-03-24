FROM tensorflow/tensorflow:2.7.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list 
RUN apt-get update && apt-get install -y --no-install-recommends \
	  apt-utils \
	  build-essential \
	  libsndfile-dev \
    llvm-9-dev \
    llvm-9-runtime \
    llvm-9-tools \
    libedit-dev \
    zsh \
    tmux \
    wget \
    git \
    cmake && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install ipython numba==0.51.2 llvmlite==0.34.0 numpy==1.19.5 && \
    pip install tensorflow-addons==0.15.0 && \
    pip install soundfile pyopenjtalk flask requests && \
    git clone https://github.com/TensorSpeech/TensorFlowTTS.git && \
    cd TensorFlowTTS && \
    pip install .

WORKDIR /app

COPY flask_app_voice.py /app/

RUN mkdir static && \
    mkdir templates

COPY index.html /app/templates/

CMD ["python3", "flask_app_voice.py"]

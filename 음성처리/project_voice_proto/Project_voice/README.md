# 프로젝트 소개

- 프로젝트 이름: Text to Voice

- 프로젝트 설명: 
    - 본 프로젝트는 시각장애인을 위한 고지서 읽어주기 서비스 우리 알리미 'SORI' 의 하위 프로젝트입니다.

    - 본 프로젝트에서는 TensorflowTTS model 을 이용하여 OCR 단계에서 읽어온 텍스트를 읽어주는 것을 목표로 합니다. 

    - 본 프로젝트에서 사용하는 TensorflowTTS model 은 TacoTron2 를 기반으로 만들어진 TTS 프로그램으로, decoder 로는 mel-GAN 을 사용하고 있습니다. 

    - TensorflowTTS 가 사용하는 라이브러리들이 현재 환경과 맞지 않는 경우가 많고, 이에 따라 Tensorflow 팀이 제공하는 기존의 Dockerfile 이 무력화되었습니다. 

    - 따라서 이를 실행하기 위해 준비과정이 너무 번거로웠기 때문에 새롭게 이를 사용자 여러분이 Dockerfile 과 app 만으로 간단하게 실행할 수 있도록 조정하는 것을 목표로 했습니다.

# 프로젝트 설치

## Prerequisites
프로젝트를 실행하기 위해서는 다음과 같은 사전 설치가 필요합니다.

- Docker

- WSL Or Linux (본 환경에서는 WSL2:Ubuntu 22.04 LTS 로 구축했습니다.)

- Dockerfile

- app.py

- index.html 

### 구현 환경

- Window 11

- WSL2: Ubuntu 22.04 LTS

- Python>=3.8 

- GPU: GeforceRTX 3080ti

- CPU: AMD Ryzen 7 5800X 8-Core Processor

- RAM == 64.0GB

## 설치 방법

- git clone 을 통해 Dockerfile 과 app 파일을 다운로드 받으셔도 좋고, 따로 파일을 다운로드 받으셔도 좋습니다. 

- Dockerfile 과 app 파일을 같은 디렉터리 안에 위치시켜 주신 뒤, 'sudo docker build -t 이미지명 .(현재 경로)' 을 터미널에서 실행시켜주세요. 정상적으로 실행이 완료되기까지 2 분 이상의 시간이 소요될 수 있습니다.

- 도커 이미지 구축이 완료되었으면 'sudo docker run -d -p 8080:8080 이미지명' 을 통해 컨테이너를 실행해주세요. 

- 포트를 다르게 열어주고 싶으시다면 py 파일을 수정하여 포트 번호를 수정할 수 있습니다. 

## 실행 방법

1. 웹 브라우저를 열고 http://localhost:8080을 입력합니다.

2. 입력란에 텍스트를 입력하고 '음성 합성' 버튼을 클릭합니다.

3. 몇 초 후 오디오 파일이 생성됩니다. 사이트에서 직접 재생하거나 Download Audio 버튼을 클릭하여 파일을 다운로드합니다.


# 예제


![웹사이트 구현](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbIYsVf%2Fbtr5HqfTkAN%2FWUXR42i04z27ZGRgO7jks0%2Fimg.png)



# 개발자 정보
- 이름: 고준성
- 이메일: rhwnstjd2004@gmail.com
- 깃허브: [링크](https://github.com/KO-JUNSUNG)


# 참조한 레퍼런스

- [1.](https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers)
- [2.](https://joungheekim.github.io/2021/04/02/code-review/)
- [3.](https://github.com/ttop32/coqui_tts_korea)
- [4.](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS?fbclid=IwAR3oyEWkgYuG2LLhQPZfkkFKnFJRNyGDA2Za1C_DYpmYvfRf8WQaGDH-xNA)
- [5.](https://pyrasis.com/tts/2023/02/05/FastSpeech2-My-Voice-TTS#%ED%95%84%EC%88%98-%EB%8D%B0%EC%9D%B4%ED%84%B0%5D)
- [6.](https://github.com/pyrasis/Korean-FastSpeech2-Pytorch)
- [7.](https://github.com/TensorSpeech/TensorFlowTTS)
- [8.](https://liusongxiang.github.io/diffsvc/)
- [9.](https://github.com/NVIDIA/tacotron2)
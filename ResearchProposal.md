# SMamba (working title)

# Introduction

- Why this research?
  - Current Text to Speech models are based on Autoregressive(AR), Non-Autoregressive(NAR), Diffusion models. And all of them are based on the basic model, the Transformer.
  - But there is a new basic model, released in December 2023, 'Mamba'. The Linear-Time Sequence Modeling with Selective State Spaces
  - There is a lot of evidence that it works well in other domains, but no evidence yet in speech synthesis. So this can be a promising challenge.
  - The architecture of Mamba is similar to Autoregressive, so at the point of replacing foundation model, Transformer to Mamba, I expect more naturalness than current work. 
  - Also, this can be a study that will lead to a lot of new research.


# Background

- Neural Codec Language Model(AR Model)
  - Advantage
    - Good at generalization, great at in-context learning(It means it needs less time for fine tuning)
  
  - Disadvantage 
    - Due to auto-regressive generative way, it has slow inference speed and lack of robustness, resulting in repetition, skipping and mispronunciation.
    - It is highly dependent on the pre-trained neural audio codec or discrete speech unit.
    - It requires a large data set to train the model.

- NAR Model
  - Advantage
    - Fast inference, training speed (O(n) -> O(1))
    - Robust to long utterances

  - Disadvantage
    - Lack of diversity because it generates the whole sentence in parallel (averaging problem)
    - lower audio quality

- Diffusion Model
  - Advantage
    - Fluent and natural
    - High performance in adaptation

  - Disadvantage 
    - Slow inference speed for its iterative generation method


- Transformer

![Transformer](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbty7Lc%2FbtsEhWeNKS1%2FuVbMovBxd4NK7HwQMQOR70%2Fimg.png)
  - Advantage:
    - Parallel processing by using attention mechanism (can process information for each position in the input sequence simultaneously)
    - Modular structure: Allows easy addition or removal of components, making it adaptable to different tasks and allowing easy customization for specific domains
  - Disadvantage
    - Dependence on large amount of data
    - Inefficiency for very long sequences


- 트랜스포머 구조에 대한 설명을 추가. 장점은 어디서로부터 기원하며, 단점은 어디서로부터 기원하는지.
 

- Mamba 
  - Mamba is the new architecture of foundation model that can replace Transformer.
  - It can overcome the deficiency of Transformer, the inefficiency on long sequences.
    1. By allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token.
    2. This change prevents the use of efficient convolutions, so the authors design a hardware-aware parallel algorithm in recurrent mode: selective SSMs in a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba).
    3. Mamba has 5x faster fast inference speed than Transformer
    4. And, Transformer needs a large amount of data for good performance, but Mamba does not. It can save the resource.

  - But it's not proven yet whether it's good at actual scale of model (over 7B parameters). 

- 맘바가 어떤 구조이고, 왜 이런 성능을 낼 수 있는지에 대한 설명과 이미지 파일을 추가
- 어떤 이유 때문에 맘바가 트랜스포머보다 빠른지에 대해서 상세한 설명이 필요함

# Research objectives and questions

- __Using 'Mamba' instead of 'Transformer' works well on Non Autoregressive based zero-shot learning Text-to-Speech?__

# Methodology

- Based on model 'VITS', conditional variational autoencoder TTS system,try adaptation and transfer learning

![VITS](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbkQQCP%2FbtsElKqKTGR%2FJK41BqvqqcK5TRCxeKKYPK%2Fimg.png)

- In HierSpeech++, they trained the model with a batch size of 160 for 1,000k stemps on eight NVIDIA A6000 GPUs.
- LibriTTS, Libri-light, and multi-speaker speech synthesis datasets on AIHub 


# Expected results

- Overcome the disadvantage of AR models (slow inference) and get the advantage (good quality, natural).
- Currently HierSpeech++'s UTMOS score is 4.16 and WER is 3.17. 
- The research goal is to get close to or better UTMOS score than current work.
  - This will provide new insights for speech synthesis. Not only Transformer architecture, but also Mamba architecture can be tried.

# Research schedule

- Undetermined

# References

- Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee. "HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-Shot Speech Synthesis.", arXiv:2311.12454 (2023).

- Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia. "Attention Is All You Need." arXiv preprint arXiv:1706.03762 (2017).

- Albert Gu ,Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752 (2023)
- Kim, Jaehyeon, Jungil Kong, and Juhee Son. "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." arXiv preprint arXiv:2106.06103 (2021).
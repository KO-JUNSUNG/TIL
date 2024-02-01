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
  - Advantage:
    - Parallel processing by using attention mechanism (can process information for each position in the input sequence simultaneously)
    - Modular structure: Allows easy addition or removal of components, making it adaptable to different tasks and allowing easy customization for specific domains
  - Disadvantage
    - Dependence on large amount of data
    - Inefficiency for very long sequences
 

- Mamba 
  - Mamba is the new architecture of foundation model that can replace Transformer.
  - It can overcome the deficiency of Transformer, the inefficiency on long sequences.
    1. By allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token.
    2. This change prevents the use of efficient convolutions, so the authors design a hardware-aware parallel algorithm in recurrent mode: selective SSMs in a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba).
    3. Mamba has 5x faster fast inference speed than Transformer
    4. And, Transformer needs a large amount of data for good performance, but Mamba does not. It can save the resource.

  - But it's not proven yet whether it's good at actual scale of model (over 7B parameters). 

# Research objectives and questions

- __Using 'Mamba' instead of 'Transformer' works well on diffusion based zero-shot learning Text-to-Speech?__

# Methodology

- Based on SOTA model under diffusion models, try adaptation and transfer learning
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

- Remaining SPEECH: A Scoping Review of Deep Learning-powered Speech Conversion
- SH Lee, HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-Shot Speech Synthesis, IEEE
- Ashish Vaswani, Attention Is All You Need, Google Brain
- Albert Gu and Tri Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, Machine Learning Department Carnegie Mellon University
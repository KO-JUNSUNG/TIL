
# 수요일까지의 목표


__= 내가 문제를 확실히 이해했다는 ppt 를 작성__

- 현재 마주한 문제는 이렇다.
- 문제를 보다 구체화하여 세부적으로 접근한다면, [1,2,3] 의 문제가 있다.
- x,y,z 방식이 있는데, 이 중 x로 접근할 것이다. 
- 그 이유는 x` 한 이유에서 근거한다. 
- 예상되는 문제점은 이와 같다.
- 이를 방비하기 위해 이런 대안을 내놓을 것이다. 
- 앞으로의 계획은 이러하다.
- 과제제안서를 바탕으로 용어를 정리하고, 생각을 develop 해서 SOM을 이렇게 사용해보겠다는 방향으로 하면 될 듯? 

# Flow

- 월요일까지 ppt를 완성
- 화요일에 발표를 준비, 예상 질문을 작성하고 부족한 부분을 보충
- 수요일에 발표

# 과제 정의

- 절대좌표(GPS)가 없는 상태에서, drone들이 어떻게 해야 서로의 상대적 거리 정보만을 가지고 "특정 형태의 대형(ex:삼각뿔)"을 만들고 유지할 수 있을까?

- __Keyword: UAV formation keeping, Formation keeping, Formation control__

## 과제를 위해 필요한 세부 문제 정의

1. Formation keeping 을 위해 사용 가능한 주요 알고리즘

![드론 제어방식](https://post-phinf.pstatic.net/MjAxODA0MTBfMTk2/MDAxNTIzMzQ2OTM0MzA5.-Fa0xf_x-T5MG4GTGQ3a_zz-hyvnCzcSsf0L8ogcj1Ig.DUQ8akgKlqjCTVIGFD5VV2A_8e8p_8r6WCg9K9WnAMgg.JPEG/10.jpg?type=w1200)
이미지 출처: [한국방위산업진흥회](https://post.naver.com/viewer/postView.nhn?volumeNo=14803929&memberNo=38486222)

   - 각 알고리즘의 장단점

| |Leader-follower| Virtual Structure| Behavior|
|------------|---------------|------------------------|---------|
|Pros|1. Simple & easy to operate <br>2. Strong at obstacle avoidence | 1. High Robustness <br> 2. Convenient to adjust the number of UAVs|1. Does not involve any central coordination <br> 2. Less computing power|
|Cons|High dependency on the leader|rigid Virtual: low turning performance of UAVs<br> flexible Virtual: poor real-time performance and accuracy|Does not guarantee the stability of the formation|

<!--
Virtual Structure: easy to set parameteres to complete the obstacle avoidence

Behavior based: difficult to define the overall formation behavior and to obtain the accurate mathematical description -->
![Virtual Structure](https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs42405-019-00180-7/MediaObjects/42405_2019_180_Fig1_HTML.png)

1. 입력정보 정의
   1. 드론들 사이의 상대적인 거리
      1. 거리 정보가 품고 있는 정보
      2. 드론들 사이의 상대적인 거리는 어떻게 측정되는가?
         - UWB 센서를 사용하여 상대적 위치를 측위(연구제안서 9 페이지)
      3. 상대적 거리를 측정하는 데 어떤 변수들이 사용되는가?
         - $P_{i}$ = 수신신호강도
         - $\phi_{i}$ = 방위각
         - $\alpha_{i}$ = 고도각 

2. 입력 정보의 특성
3. 특성을 고려한 최적의 방법


## 용어 정의

- 거리 정보 = 수신 신호 강도(RSS; Received Signal Strength) & 도달 각도(AOA; Arrival of Angle) = $\hat{P_{i}}, \hat{\phi_{i}}, \hat{\alpha_{i}}$
- Relative distance: 드론들간의 상대적인 위치 추정에 필요한 parameter들을 이하에 정리



## 참조 레퍼런스

<!-- Leader follower, 2020 년-->
[[1]](https://doi.org/10.1016/j.cja.2019.08.009) Guibin SUN, Rui ZHOU, Kun XU, Zhi WENG, Yuhang ZHANG, Zhuoning DONG, Yingxun WANG,
"Cooperative formation control of multiple aerial vehicles based on guidance route in a complex task environment",
Chinese Journal of Aeronautics,
Volume 33, Issue 2,
2020,
Pages 701-720,
ISSN 1000-9361, https://doi.org/10.1016/j.cja.2019.08.009.

<!-- virtual structure 2022 년-->
[[2]](https://doi.org/10.1016/j.oceaneng.2022.111148) Qingzhe Zhen, Lei Wan, Yulong Li, Dapeng Jiang,
"Formation control of a multi-AUVs system based on virtual structure and artificial potential field on SE(3)",
Ocean Engineering,
Volume 253,
2022,
111148,
ISSN 0029-8018, https://doi.org/10.1016/j.oceaneng.2022.111148.


<!-- Leader follower 2023 년-->
[[3]](https://doi.org/10.1016/j.cja.2023.07.030) Hanlin SHENG, Jie ZHANG, Zongyuan YAN, Bingxiong YIN, Shengyi LIU, Tingting Bai, Daobo WANG,
"New multi-UAV formation keeping method based on improved artificial potential field",
Chinese Journal of Aeronautics,
Volume 36, Issue 11,
2023,
Pages 249-270,
ISSN 1000-9361, https://doi.org/10.1016/j.cja.2023.07.030.

<!-- leader follower 2014년-->

[[4]](https://ieeexplore.ieee.org/document/6858777) “Consensus-based cooperative formation control with collision avoidance for a multi-UAV system | IEEE Conference Publication | IEEE Xplore,” ieeexplore.ieee.org. https://ieeexplore.ieee.org/document/6858777 (accessed Feb. 08, 2024).
‌
<!-- behavior 2014년-->

[[5]](https://ieeexplore.ieee.org/iel7/5962385/6104215/06646274.pdf?casa_token=UzM-82WG2MUAAAAA:_NBtv5QsIE93RnNP9tNrZqKkVSq_d6oXRqlq6cXD6vvjodEHoIJgjwT9taQNqb7X_8-aypDMpQ) J.L. Lin, K.S. Hwang, Y.L. Wang
"A simple scheme for formation control based on weighted behavior learning"
IEEE Trans Neural Netw Learn Syst, 25 (6) (2014), pp. 1033-1044
DOI: 10.1109/TNNLS.2013.2285123

<!--Virtual Sturcutre-->
__[[6]](https://doi.org/10.1007/s42405-019-00180-7) Zhang, B., Sun, X., Liu, S. et al. "Formation Control of Multiple UAVs Incorporating Extended State Observer-Based Model Predictive Approach." Int. J. Aeronaut. Space Sci. 20, 953–963 (2019). https://doi.org/10.1007/s42405-019-00180-7__


<!-- 목표와 가장 비슷해 보이는 논문, leader follower 베이스 논문 같음-->
__[[7]](https://doi.org/10.1007/978-3-031-19759-8_7) Brandstätter, A., Smolka, S.A., Stoller, S.D., Tiwari, A., Grosu, R. (2022). Towards Drone Flocking Using Relative Distance Measurements. In: Margaria, T., Steffen, B. (eds) Leveraging Applications of Formal Methods, Verification and Validation. Adaptation and Learning. ISoLA 2022. Lecture Notes in Computer Science, vol 13703. Springer, Cham. https://doi.org/10.1007/978-3-031-19759-8_7__

# 해야하는 일

- [] 상대거리를 어떻게 측정하는지에 대해 제대로 학습하기 (사용 용어, 정의 기준)
- [] formation keeping에 대해 각각의 알고리즘이 어떻게 작동하는지에 대해 설명하기

<!-- 
## Additional information

## SOM
![SOM](http://i.imgur.com/eHUVAtr.gif)
- SOM 알고리즘은 일종의 PCA 이자 sorting 알고리즘이다. 
- 노드의 그리드로 구성되며, 각 노드에는 입력 데이터셋과 동일한 차원의 가중치 벡터가 포함되어 있다. 무작위로 초기화될 수 있지만, 사전분포가 적절할 경우 학습속도가 빨라진다. 

### SOM 알고리즘 진행 절차

- 모든 가중치 벡터의 데이터 공간 상에서 유클리디언 거리를 계산해 가장 좋은 노드인 BMU를 찾는다. 입력 벡터쪽으로 업데이트하면서 이웃 노드도 일부 계속해서 조정되는데, 이 이웃노드의 이동 정도는 neighborhood function 에 의해 결정된다. 
- 네트워크가 완전히 converge 할 때까지 샘플링을 사용해 여러 차례 반복적으로 이루어진다. 
- SOM을 가지고 Localization 을 어떻게 하는가? 
  - (summary of paper: Wireless Localization Using Self-Organizing Maps) -->








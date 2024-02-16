# 과제 정의

- 절대좌표(GPS)가 없는 상태에서, drone들이 어떻게 해야 유클리드 거리 정보만을 가지고 "특정 형태의 대형(ex:삼각뿔)"을 만들고 유지할 수 있을까?

- __Keyword: UAV formation keeping, Formation keeping, Formation control__

## 과제를 위해 필요한 세부 문제 정의

1. Formation keeping 을 위해 사용 가능한 주요 알고리즘

![드론 제어방식](https://post-phinf.pstatic.net/MjAxODA0MTBfMTk2/MDAxNTIzMzQ2OTM0MzA5.-Fa0xf_x-T5MG4GTGQ3a_zz-hyvnCzcSsf0L8ogcj1Ig.DUQ8akgKlqjCTVIGFD5VV2A_8e8p_8r6WCg9K9WnAMgg.JPEG/10.jpg?type=w1200)
이미지 출처: [한국방위산업진흥회](https://post.naver.com/viewer/postView.nhn?volumeNo=14803929&memberNo=38486222)

2. 각 알고리즘의 장단점

| |Leader-follower| Virtual Structure| Behavior|
|------------|---------------|------------------------|---------|
|Pros|1. Simple & easy to operate <br>2. Strong at obstacle avoidence | 1. High Robustness <br> 2. Convenient to adjust the number of UAVs|1. Does not involve any central coordination <br> 2. Less computing power|
|Cons|High dependency on the leader|rigid Virtual: low turning performance of UAVs<br> flexible Virtual: poor real-time performance and accuracy|Does not guarantee the stability of the formation|

<!--
Virtual Structure: easy to set parameteres to complete the obstacle avoidence

Behavior based: difficult to define the overall formation behavior and to obtain the accurate mathematical description -->

3. 결정된 모델

![Virtual Structure](https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs42405-019-00180-7/MediaObjects/42405_2019_180_Fig1_HTML.png)


## 결정된 모델, Virtual Structure 을 구현하는 방법

1. Self organizing Map 방법
2. Artificial potential field 방법


## 용어 정의

- 거리 정보 = 

# 해야하는 일

- [] 상대거리를 어떻게 측정하는지에 대해 제대로 학습하기 (사용 용어, 정의 기준)
- [] formation keeping에 대해 각각의 알고리즘이 어떻게 작동하는지에 대해 설명하기





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

<!--Virtual Sturcutre, 28 citations 96 percentile-->
__[[6]](https://doi.org/10.1007/s42405-019-00180-7) Zhang, B., Sun, X., Liu, S. et al. "Formation Control of Multiple UAVs Incorporating Extended State Observer-Based Model Predictive Approach." Int. J. Aeronaut. Space Sci. 20, 953–963 (2019). https://doi.org/10.1007/s42405-019-00180-7__


<!-- Leader-follower, SOM algorithm, 43 citations 93th percentile-->
[[7]](https://doi.org/10.1016/j.oceaneng.2020.108048)Yan-Li Chen, Xi-Wen Ma, Gui-Qiang Bai, Yongbai Sha, Jun Liu,
Multi-autonomous underwater vehicle formation control and cluster search using a fusion control strategy at complex underwater environment,
Ocean Engineering,
Volume 216,
2020,
108048,
ISSN 0029-8018,
https://doi.org/10.1016/j.oceaneng.2020.108048.
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

<!--
$\left\{\begin{matrix}
 \dot{p_{i}} = u_{i}\\
\dot{u_{i}} = \varrho_{i}
\end{matrix}\right. ,(8)$

$p_{i},u_{i}$ are the position and velocity of the ith particle in the inertial frame. $\varrho_{i}$ are the control input of the ith particle.

$\varrho_{i} = c_{1}(p_{ri}-p_{i}) + c_{2}(u_{ri}-u_{i}) + F_{i} ,(9)$

$c_{1}$ is the position error feedback coefficient, $c_{2}$ is the speed error feedback coefficient. $p_{ri}$ and $u_{ri}$ are respectively the position and speed of the ith mass point correspoonding to the expected virtual structure in the inertial frame. $F_{i} = -\triangledown_{qi} \mathfrak{V}(q)$ is the negative gradient term of the artificial potential field to realize collision avoidance between particles. Simplify $F_{i}$ to 0, the simplified control input is

$\varrho_{i} = c_{1}(p_{ri}-p_{i}) + c_{2}(u_{ri}-u_{i}), (10)$

Substitute Eq.(10) into Eq.(8), we get (11).

$\ddot{p_{i}} + c_{2}\dot{p_{i}} + c_{1}p_{i} = c_{2}\dot{p_{ri}} + c_{1}\dot{p_{ri}}, (11)$

It appears to be a lninear second-order system. To make the system stable, we should make $c_{1}>0$, $c_{2}<0$ according to the Hurwite criterion.


The potential field force is only used for collision avoidance, so the design of the artificial potential filed can be simplified. Considering that the particle i is in the potential filed generated by other particles nearby, the total potential energy of the ith particle can be expressed as 

$\mathfrak{V}(p) = \sum_{j,i=1}^{N}\zeta(||p_{j} - p_{i}||), j\neq i (12)$

$\phi(x)$ is the potential energy function, and $||\cdot||$ is the modulus of the vector. That is the artificial potential field energy is a function of the relative distance of the particles. Since only considering the artificial potential field for collision avoidance, rather than organizing the formation, only the potential field is required to have repulsive force. Construct potential function 

$\zeta(z) =\left\{\begin{matrix}
\frac{1}{2}k(\frac{1}{z} - \frac{1}{R}) &, 0 < z \leq R  \\
0 &  ,z>R\\
\end{matrix}\right. (13)$

k is the potential field strength coefficient and its selection should consider the inertia of the AUV and the control output of the actuator; if k is too small, it is not conducive to safe collision avoidance; if k is too large, it may exceed the output range of the actuator which has no practical meaning; R is the range of potential field.


In order to avoid interference of the artificial potential function on the formation, the distance between the particles in the final formation should be greater than the range of potential function, namely

$||p_{j}-p_{i}||\geq R   ,\forall (i,j) \in N, j\neq i (14)$
The artificial potential field effect in the formation process can be regarded as interference to the second-order system. The existence of the negative gradient term of the artificial potential field causes the particle to move in the direction where the energy of the artificial potential field is reduced, thereby avoiding collision.

Compared with the artificial potential field method, the virtual structure method is more conducive to the organization of formation.
This paper is designed to generate and transform the formation after the AUV reaching the specified depth, so the position and speed of the simplified formation reference point in the system are expressed as $p_{r} \in \mathbb{R}^{2}$, $u_{r} \in \mathbb{R}^{2}$. We have


$\left\{\begin{matrix}
p_{ri}&p_{r} + R(\theta)\mathfrak{L}r_{i}  \\
u_{ri}&u_{r} + \dot{R}(\theta)\dot{\mathfrak{L}}r_{i}  \\
\end{matrix}\right. (15)$

where $R(\theta) = \begin{bmatrix}
cos\theta &  -sin\theta\\
sin\theta &  cos\theta\\
\end{bmatrix}$ represents the formation rotation matrix of the virtual structure; $\mathfrak{L} = \begin{bmatrix}
l &  0\\
0 &  l\\
\end{bmatrix}$ is the formation zoom factor; $r_{i}$ is the psotion of the ith mass point relative to the formation reference constant, the formation shape remains constant; if $\theta$ is expressed as the direction of the velocity vector of the formation reference point, the attitude of the formation is remain unchanged. Since the collision is adopted for the change of the formation shape, that is $r_{i} \to r_{i}^{'}$, We avoid the design of $\dot r_{i}$ which simplifies the formation change.
-->

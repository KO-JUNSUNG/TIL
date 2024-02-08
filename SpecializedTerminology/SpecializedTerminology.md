## Receptive Field

- (convolution) neuron 하나가 표현하는 데이터의 일부. 혹은 뉴런이 입력 데이터에서 영향을 받는 지역이라고도 한다.
- Filter의 크기가 클수록, 네트워크가 깊어질수록 수용필드의 크기가 커진다. 
- Receptive field가 크면 이미지 혹은 데이터의 전체적인 특성을 보게 되고 보다 많은 연산을 필요로 하게 된다. 반대로 receptive field가 작다면 보다 local 한 특징에 집중하게 된다. 

[Receptive Field](https://hyunhp.tistory.com/695)


## 분류는 어떤 함수를 쓰는 게 좋나요?

- 복잡한 이진 분류에는 소프트맥스를 하는 것이 좋습니다.
- 대부분의 이진분류에서는 시그모이드를 사용한 단일 출력이 더 간단하고 좋죠.
- 뉴런 하나 쓸래? 두 개 쓸래?
# Business-Analytics-Tutorial
2022년 2학기 BA

![image](https://user-images.githubusercontent.com/72682160/195506768-899e573f-ef46-4f99-8c34-686101c2cee1.png)

supervised selection: GA
unsupervised extraction: t-SNE

## t-SNE
t-SNE는 Nonlinear embedding을 사용하는 unsupervised extraction 방법론으로 최근 고차원 데이터의 시각화에 가장 보편적으로 사용되는 방법론이다.
t-SNE는 Stochastic Neighbor Embedding(SNE)과 Symmetric SNE를 순서대로 거쳐 발전한 알고리즘이기 때문에 각각을 우선 살펴 보도록 하자.

### SNE(Stochastic Neighbor Embedding)
핵심적인 아이디어는 Locally Linear Embedding(LLE)와 동일하게 "본래의 고차원에서의 이웃간 의 관계와 저 차원으로  Embedding된 후 이웃관의 관계가 보존되어야한다"는 것이나, Local 이웃 간의 거리가 확정적(deterministic)이  아닌  확률적(probabilistic)으로 정의가 된다.

즉 방법론의 이름 처럼 "Stochastic"하게 "Neighbor"를 활용해 축소된 저차원의 "Embedding"를 얻는다고 이해하면 좋을 것이다.
이 과정을 그림으로 표현하면 다음과 같다.
![image](https://user-images.githubusercontent.com/72682160/195508426-4b975f71-bf6b-4be2-9935-a19357605e38.png)
LLE(좌측): 만약 중심의 데이터 포인트 P를 복원(표현)하기 위해 이웃하는 데이터 포인트 6개를 사용한다면 그림처럼 P와 가장 가까운 6개의 점이 바로 확정적으로 결정이 된다.

SNE(우측): 그러나 SNE의 경우에는 중심의 데이터 P를 표현하기 위해 이웃하는 점을 사용한다는 것은 동일하나, Local 이웃 간의 거리가 확률적(probabilistic)으로 정의되기 때문에 그림처럼 1에서 6의 가까운 점이 확정적으로 사용되는 것이 아니라, 그보다 멀리 있는 a~f 점도 함께 사용될 가능성이 존재하게 된다.


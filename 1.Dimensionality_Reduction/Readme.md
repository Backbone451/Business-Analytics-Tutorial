- [Dimensionality Reduction](#dimensionality-reduction)
  * [Locally Linear Embedding(LLE)](#locally-linear-embedding)
  * [t-SNE](#t-sne)
    + [SNE(Stochastic Neighbor Embedding)](#sne-stochastic-neighbor-embedding)
  * [Reference](#Reference)

# Dimensionality Reduction
&nbsp;이번 Tutorial에서는 데이터의 차원이 커질 수록 발생하는 Curse of dimensionality 문제를 해결하기 위한 Dimensionality Reduction 방법론들 중 대표적인 Nonlinear unsupervised extraction 방법론 LLE와 t-SNE에 대해 살펴보도록 하자.

![image](https://user-images.githubusercontent.com/72682160/195506768-899e573f-ef46-4f99-8c34-686101c2cee1.png)

&nbsp;본 글에서는 다음의 두 방법론에 대한 이론적 내용을 간단히 소개하고, 각각의 notebook file을 통해 python code 실습을 수행한다.  
unsupervised extraction: LLE, t-SNE  

## Locally Linear Embedding
&nbsp;LLE의 기본적인 아이디어는 Local(지역적) 정보의 보존이 Global(전역적) 정보의 보존 보다 중요하다는 가정아래 각각의 데이터 포인트를 이웃하는 데이터 포인트 들과의 관계를 통해 표현하는 것으로 시작된다.
![image](https://user-images.githubusercontent.com/72682160/195516365-971cf668-83bd-499c-a6a3-d635a1ad25c0.png)
&nbsp;위 예시 그림을 보면 Representation을 얻고자하는 중심 데이터 포인트(붉은색)이 있고, 그 주변의 데이터 포인트(푸른색) 확인할 수 있다.  
&nbsp;만약 우측의 그림에서처럼 중심이 되는 점과 이웃하는 점들 사이의 관계를 잘 파악한다면 그들을 통해 중심의 점을 표현하는 것이 가능할 것이다.  

![image](https://user-images.githubusercontent.com/72682160/195516903-82fd0f92-4e12-4636-b395-299c5ba2e87e.png)

&nbsp;즉, 이러한 관계를 충분히 학습하여 기억하고 있다면 모든 점들이 함께 저차원으로 이동하더라도 점들 사이의 관계는 유지될 것 이므로 위 그림처럼 차원에서도 동일한 이웃 점들과 동일한 관계를 통해 중심의 점이 충분히 표현될 수 있다.  

전체적인 과정을 정리하면 다음과 같다.  
### Step 1. 각 데이터 포인트의 이웃을 할당한다.  

### Step 2. 할당된 이웃들을 활용하여 중심의 데이터를 표현하기 위한 최적의 가중치를 찾는다.  

<image src="https://user-images.githubusercontent.com/72682160/195541483-de3a9a08-c1b0-4bea-8e48-fea5eab584dc.png" height="200"/>
&nbsp;이 과정 자체에서 한번의 최적화가 필요하며, 위 식의 E(W)가 cost function으로 최소화 대상이다.  

&nbsp;&nbsp;E(W): 가중치 행렬이 W일때, 이웃으로 표현한 데이터 x_𝑖와 x_𝑗 사이의 차이를 제곱한 것  

&nbsp;&nbsp;x_𝑖: 중심이 되는 데이터 포인트  

&nbsp;&nbsp;x_𝑗: x_𝑖의 이웃이 되는 데이터 포인트  

&nbsp;&nbsp;𝑊_𝑖𝑗: x_𝑖와 x_𝑗사이의 가중치  

&nbsp;두 가지의 제약식은 각각 이웃이 아닌 경우에는 가중치가 0이고, 이웃 점들의 가중합으로 중심을 복원하는 형태이기 때문에 한 중심 데이터 포인트에 대한 가중치의 합은 1인것을 의미한다.  

&nbsp;즉, 최적화를 통해 Local 정보를 활용해 각각의 데이터를 표현하는 최적의 가중치 행렬을 학습하는 것이다.  

### Step 3. 새로 얻은 가중치 행렬 𝑊_𝑖𝑗 를 활용하여 축소된 차원으로 데이터를 표현한다.
<image src="https://user-images.githubusercontent.com/72682160/195543877-2656b66e-436b-40af-911b-6af4735b94e9.png" height="60"/>
&nbsp;위 식에서 보이는 것처럼 동일한 가중치 행렬 𝑊_𝑖𝑗 를 활용하여 축소된 차원에서도 Local 정보를 통한 데이터의 복원이 유효하도록 하는 representation y를 찾는 과정이다.  
&nbsp;이 식을 정리하면 다음과 같이 정리할 수 있다.   
<image src="https://user-images.githubusercontent.com/72682160/195545616-93459751-185d-4275-8377-ab0af74b09fe.png" height="310"/>  

&nbsp;&nbsp;Φ(W): 가중치 행렬이 W일때, 이웃으로 표현한 데이터 y_𝑖와 y_𝑗 사이의 차이를 제곱한 것  

&nbsp;&nbsp;y_𝑖: 축소한 저차원에서 중심이 되는 데이터 포인트  

&nbsp;&nbsp;y_𝑗: y_𝑖의 이웃이 되는 데이터 포인트  

&nbsp;&nbsp;𝑊_𝑖𝑗: 원래의 차원에서 얻은 x_𝑖와 x_𝑗사이의 가중치  

&nbsp;여기서 두 가지의 제약식은 그림에 나타난 것과 같이 Embedding 공간에서 각 변수의 평균을 0으로 만들고, 각 변수들을 직교하게 하는 것으로 새로운 좌표계를 만든 것이라고 이해할 수 있다.  


## t-SNE
&nbsp;t-SNE는 Nonlinear embedding을 사용하는 unsupervised extraction 방법론으로 최근 고차원 데이터의 시각화에 가장 보편적으로 사용되는 방법론이다.  

&nbsp;t-SNE는 Stochastic Neighbor Embedding(SNE)과 Symmetric SNE를 순서대로 거쳐 발전한 알고리즘이기 때문에 각각을 우선 살펴 보도록 하자.  


### SNE(Stochastic Neighbor Embedding)
&nbsp;핵심적인 아이디어는 Locally Linear Embedding(LLE)와 동일하게 "본래의 고차원에서의 이웃간 의 관계와 저 차원으로  Embedding된 후 이웃관의 관계가 보존되어야한다"는 것이나, Local 이웃 간의 거리가 확정적(deterministic)이  아닌  확률적(probabilistic)으로 정의가 된다.  

&nbsp;즉 방법론의 이름 처럼 "Stochastic"하게 "Neighbor"를 활용해 축소된 저차원의 "Embedding"를 얻는다고 이해하면 좋을 것이다.  
&nbsp;이 과정을 그림으로 표현하면 다음과 같다.  
![image](https://user-images.githubusercontent.com/72682160/195508426-4b975f71-bf6b-4be2-9935-a19357605e38.png)
&nbsp;&nbsp;LLE(좌측): 만약 중심의 데이터 포인트 P를 복원(표현)하기 위해 이웃하는 데이터 포인트 6개를 사용한다면 그림처럼 P와 가장 가까운 6개의 점이 바로 확정적으로 결정이 된다.  

&nbsp;&nbsp;SNE(우측): 그러나 SNE의 경우에는 중심의 데이터 P를 표현하기 위해 이웃하는 점을 사용한다는 것은 동일하나, Local 이웃 간의 거리가 확률적(probabilistic)으로 정의되기 때문에 그림처럼 1에서 6의 가까운 점이 확정적으로 사용되는 것이 아니라, 그보다 멀리 있는 a~f 점도 함께 사용될 가능성이 존재하게 된다.  

## Reference
https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7 (고려대학교 산업경영공학과 강필성 교수님 Business-Analytics 수업자료)

- [Dimensionality Reduction](#dimensionality-reduction)
  * [1. Locally Linear Embedding](#1-locally-linear-embedding)
    + [Step 1. 각 데이터 포인트의 이웃을 할당한다.](#step-1--------------------)
    + [Step 2. 할당된 이웃들을 활용하여 중심의 데이터를 표현하기 위한 최적의 가중치를 찾는다.](#step-2--------------------------------------------)
    + [Step 3. 새로 얻은 가중치 행렬 𝑊_𝑖𝑗 를 활용하여 축소된 차원으로 데이터를 표현한다.](#step-3-----------------------------------------------)
  * [2. t-SNE](#2-t-sne)
    + [2.1. SNE(Stochastic Neighbor Embedding)](#21-sne-stochastic-neighbor-embedding-)
    + [2.2. Symmetric SNE](#22-symmetric-sne)
    + [2.3. t-SNE](#23-t-sne)
  * [Reference](#reference)



# Dimensionality Reduction
&nbsp;이번 Tutorial에서는 데이터의 차원이 커질 수록 발생하는 Curse of dimensionality 문제를 해결하기 위한 Dimensionality Reduction 방법론들 중 대표적인 Nonlinear unsupervised extraction 방법론 LLE와 t-SNE에 대해 살펴보도록 하자.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195506768-899e573f-ef46-4f99-8c34-686101c2cee1.png" height="300"/>   
</p>

&nbsp;본 글에서는 다음의 두 방법론에 대한 이론적 내용을 간단히 소개하고, 각각의 notebook file을 통해 python code 실습을 수행한다.  

Nonlinear unsupervised extraction: LLE, t-SNE  

## 1. Locally Linear Embedding
&nbsp;LLE의 기본적인 아이디어는 Local(지역적) 정보의 보존이 Global(전역적) 정보의 보존 보다 중요하다는 가정아래 각각의 데이터 포인트를 이웃하는 데이터 포인트 들과의 관계를 통해 표현하는 것으로 시작된다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195516365-971cf668-83bd-499c-a6a3-d635a1ad25c0.png" height="300"/>   
</p>

&nbsp;위 예시 그림을 보면 Representation을 얻고자하는 중심 데이터 포인트(붉은색)이 있고, 그 주변의 데이터 포인트(푸른색) 확인할 수 있다. 만약 우측의 그림에서처럼 중심이 되는 점과 이웃하는 점들 사이의 관계를 잘 파악한다면 그들을 통해 중심의 점을 표현하는 것이 가능할 것이다.  
    
    
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195516903-82fd0f92-4e12-4636-b395-299c5ba2e87e.png" height="300"/>   
</p>


&nbsp;즉, 이러한 관계를 충분히 학습하여 기억하고 있다면 모든 점들이 함께 저차원으로 이동하더라도 점들 사이의 관계는 유지될 것 이므로 위 그림처럼 차원에서도 동일한 이웃 점들과 동일한 관계를 통해 중심의 점이 충분히 표현될 수 있다.  

전체적인 과정을 정리하면 다음과 같다.  


### Step 1. 각 데이터 포인트의 이웃을 할당한다.  

### Step 2. 할당된 이웃들을 활용하여 중심의 데이터를 표현하기 위한 최적의 가중치를 찾는다.  

<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195541483-de3a9a08-c1b0-4bea-8e48-fea5eab584dc.png" height="200"/>   
</p>

&nbsp;이 과정 자체에서 한번의 최적화가 필요하며, 위 식의 E(W)가 cost function으로 최소화 대상이다.  

* E(W): 가중치 행렬이 W일때, 이웃으로 표현한 데이터 x_𝑖와 x_𝑗 사이의 차이를 제곱한 것  

* x_𝑖: 중심이 되는 데이터 포인트  

* x_𝑗: x_𝑖의 이웃이 되는 데이터 포인트  

* 𝑊_𝑖𝑗: x_𝑖와 x_𝑗사이의 가중치  

&nbsp;두 가지의 제약식은 각각 이웃이 아닌 경우에는 가중치가 0이고, 이웃 점들의 가중합으로 중심을 복원하는 형태이기 때문에 한 중심 데이터 포인트에 대한 가중치의 합은 1인것을 의미한다.  

&nbsp;즉, 최적화를 통해 Local 정보를 활용해 각각의 데이터를 표현하는 최적의 가중치 행렬을 학습하는 것이다.  

### Step 3. 새로 얻은 가중치 행렬 𝑊_𝑖𝑗 를 활용하여 축소된 차원으로 데이터를 표현한다. 
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195543877-2656b66e-436b-40af-911b-6af4735b94e9.png" height="60"/>
</p>

&nbsp;위 식에서 보이는 것처럼 동일한 가중치 행렬 𝑊_𝑖𝑗 를 활용하여 축소된 차원에서도 Local 정보를 통한 데이터의 복원이 유효하도록 하는 representation y를 찾는 과정이다.  
&nbsp;이 식을 정리하면 다음과 같이 정리할 수 있다.  
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195545616-93459751-185d-4275-8377-ab0af74b09fe.png" height="310"/>    
</p>


* Φ(W): 가중치 행렬이 W일때, 이웃으로 표현한 데이터 y_𝑖와 y_𝑗 사이의 차이를 제곱한 것  

* y_𝑖: 축소한 저차원에서 중심이 되는 데이터 포인트  

* y_𝑗: y_𝑖의 이웃이 되는 데이터 포인트  

* 𝑊_𝑖𝑗: 원래의 차원에서 얻은 x_𝑖와 x_𝑗사이의 가중치  

&nbsp;여기서 두 가지의 제약식은 그림에 나타난 것과 같이 Embedding 공간에서 각 변수의 평균을 0으로 만들고, 각 변수들을 직교하게 하는 것으로 새로운 좌표계를 만든 것이라고 이해할 수 있다.  

&nbsp;본 최적화 과정을 바로 수행하기 전에 우선적으로 다음 그림에 나타난 식의 형태를 보고 결과를 예측해 볼수가 있다.  
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195550823-20ffb052-fde6-4ba7-b9cf-75a73e3d58cd.png" height="310"/>  
</p>


&nbsp;좌측의 정리한 최적화 식을 보면 우측의 "PCA" 방법론의 식과 매우 비슷하다는 것을 알 수 있다.  

&nbsp;식의 형태 자체는 동일하며 차이점은 PCA의 경우 Maximization 문제이고 LLE의 경우엔 minimization이라는 것이다.  

&nbsp;즉, PCA에서 최적화의 결과로 행렬 S의 eigen value를 큰 순서로 나열하고 거기서 원하는 차원의 수 만큼의 eigen vector를 통해 축소된 Principal component(축)을 찾아낸 것과 같이, LLE에서도 이웃 가중치 행렬 M의 eigen value를 "작은" 순서로 나열하고 거기서 원하는 차원의 수 만큼의 eigen vector를 선택하는 것으로 차원 축소를 달성 할 수 있을 것이다. (Rayleitz-Ritz theorem)

&nbsp;마지막으로 전체적인 과정을 나타내면 다음과 같다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195557931-b7a6a23f-dc53-4b88-9c6a-82a6c3b9111a.png" height="300"/>  
</p>
&nbsp;하나 주목할 점은 좌측의 그림에서 d+1개의 eigen vector를 선택한다는 점인데, 이는 가장 작은 eigen vector의 경우 모든성분에 대해서 unit vector이기 때문에 그를 bottom으로 깔아놓고 하나를 더 선택한다고 이해하면 된다.  


## 2. t-SNE
&nbsp;t-SNE는 Nonlinear embedding을 사용하는 unsupervised extraction 방법론으로 최근 고차원 데이터의 시각화에 가장 보편적으로 사용되는 방법론이다.  

&nbsp;t-SNE는 Stochastic Neighbor Embedding(SNE)과 Symmetric SNE를 순서대로 거쳐 발전한 알고리즘이기 때문에 각각을 우선 살펴 보도록 하자.  


### 2.1. SNE(Stochastic Neighbor Embedding)
&nbsp;핵심적인 아이디어는 Locally Linear Embedding(LLE)와 동일하게 "본래의 고차원에서의 이웃간 의 관계와 저 차원으로  Embedding된 후 이웃관의 관계가 보존되어야한다"는 것이나, Local 이웃 간의 거리가 확정적(deterministic)이  아닌  확률적(probabilistic)으로 정의가 된다.  

&nbsp;즉 방법론의 이름 처럼 "Stochastic"하게 "Neighbor"를 활용해 축소된 저차원의 "Embedding"를 얻는다고 이해하면 좋을 것이다.  
&nbsp;이 과정을 그림으로 표현하면 다음과 같다.  
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195508426-4b975f71-bf6b-4be2-9935-a19357605e38.png" height="310"/>  
</p>

&nbsp;&nbsp;LLE(좌측): 만약 중심의 데이터 포인트 P를 복원(표현)하기 위해 이웃하는 데이터 포인트 6개를 사용한다면 그림처럼 P와 가장 가까운 6개의 점이 바로 확정적으로 결정이 된다.  

&nbsp;&nbsp;SNE(우측): 그러나 SNE의 경우에는 중심의 데이터 P를 표현하기 위해 이웃하는 점을 사용한다는 것은 동일하나, Local 이웃 간의 거리가 확률적(probabilistic)으로 정의되기 때문에 그림처럼 1에서 6의 가까운 점이 확정적으로 사용되는 것이 아니라, 그보다 멀리 있는 a~f 점도 함께 사용될 가능성이 존재하게 된다.  

&nbsp;이때, 객체 i가 객체 j를 이웃으로 택할 확률은 다음과 같이 계산된다.  
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195583689-4f359f20-2e29-4041-93ea-daef1a9b97ae.png" height="250"/>  
</p>
&nbsp;위 식에서 분모의 경우는 후보가 되는 모든 객체들의 확률 합을 1로 만들어 주는 Normalization 역할을 수행하며 핵심적인 내용은 분자에 나타난다. x와 y는 각각 기존의 차원과 축소된 저차원에서 객체들의 representation이며, e에 음의 지수로 붙어있기 때문에 두 객체가 가까울 수록 전체 확률의 값이 증가하는 형태이다.  


&nbsp;여기서 추가로 고려해야하는 점은 원래의 차원에서 데이터 객체를 표현할 때, 가우시안 분포를 가정하여 σ를 분모에 활용한 것을 볼 수 있다. 앞서 설명한 바와 같이 SNE의 경우에는 이웃으로 선택하는 기준이 확률적이므로 사용하는 데이터셋 마다 데이터의 밀도가 다르면 전혀 다른 수의 이웃을 할당하게 될 가능성이 있다. 즉, 적절하게 σ를 조절하여 이웃하는 객체의 수를 일정수준으로 유지할 필요가 존재한다.  


<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195586071-99d8f134-a610-46ec-a031-91bff074d9ea.png" height="100"/>  
</p>

* 그림처럼 σ가 큰 경우 객체가 더 멀리 떨어져 있어도 이웃으로 선택될 확률이 커진다.

&nbsp;다음의 식에서 알 수 있듯이 이웃 확률에 따라 데이터의 Perplexity, Entropy가 결정이되는데, 실제로 알고리즘을 사용할 때는 우선적으로 원하는 수준의 Entropy를 결정해 놓고 그에 맞는 σ를 설정하는 식으로 사용된다. 

<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195586689-4f5f8bab-f009-409e-a4c6-fe92e69a995a.png" height="100"/>  
</p>

* 알려진 바에 따르면 SNE의 성능은  perplexity 5~50 사이에 강건

&nbsp;다음으로는 이제 위의 정의한 확률을 사용하여 정의한 저차원의 y가 얼마나 잘 embedding 되었는가를 평가하는 Cost Function을 정의해야한다.  
이는 방법론의 목적인 “본래의 고차원에서의 이웃간 의 관계와 저 차원으로  Embedding된 후 이웃관의 관계가 보존되어야 함”이 얼마나 잘 달성되었는지, 즉, 임베딩 전 후의 분포가 얼마나 동일한지를 통해 평가 될 수 있다.  

&nbsp;때문에 **Kullback-Leibler divergence**를 사용하여 다음과 같이 최적화 과정을 설계한다.  

<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195587686-c00d896b-2886-4c1e-91b8-4ab89c7d3113.png" height="250"/>  
</p>

&nbsp;조금더 설명을 덧붙이자면 Kullback-Leibler divergence loss를 현재 최적화의 대상 미지수 y에 대해 편미분하여 Gradient를 구하고 Gradient Update를 통해 cost function을 최소화하는 것이다.  

### 2.2. Symmetric SNE
&nbsp;다음으로 알아볼 방법은 기존의 SNE가 두 객체 사이에 이웃으로 선택할 확률이 다른 것을(P(i -> j) ≠ P(j -> i)) 동일하게 만든 Symmetric SNE이다.  

&nbsp;식은 다음과 같이 두 객체 각각이 서로를 이웃으로 선택할 확률을 더하고 2n으로 나눠주는 방식으로 매우 직관적이며, 여기서 2n의 제약식은 어떤 객체로 부터 또 다른 객체를 선택할 확률의 하한을 정해준 것으로 이해하면 된다.  

<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195588391-e35054da-29fe-42d7-a75f-5b3eb0eeb50f.png" height="200"/>  
</p>

&nbsp;최적화과정은 SNE와 거의 동일하게 다음과 같이 정리된다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195588638-d9dc5b09-23c8-4004-b144-aaef8841c681.png" height="200"/>  
</p>

&nbsp;기존의 SNE보다 더욱 간단하게 최적화가 수행되며 대칭성도 달성하였지만 Symmetric SNE에는 아주 가까운 거리의 객체에 비해 적당히(moderate) 떨어진 객체들이 선택될 확률이 급격하게 감소한다는 Crowding Problem이 존재한다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195589077-e3ed48cc-4caa-4c94-bf4e-124a5b32f8c8.png" height="200"/>  
</p>

&nbsp;이웃으로 선택될 확률을 나타내는 Gaussian 분포의 그림에서 초록색과 붉은색의 기울기를 보면 바로 알 수 있듯이, 이러한 Crowding Problem은 Gaussian 분포의 뾰족한 모양 때문에 발생한다.  

### 2.3. t-SNE
&nbsp;Symmetric SNE의 Crowding Problem을 해결하기 위해 Gaussian 분포를 눌러 조금더 납작하게 만든 t 분포(df=1)를 사용해 SNE를 수행하는 방법론이 바로 오늘의 주제인 t-SNE이다.  

&nbsp;t-SNE에서는 원래의 고차원에서는 Gaussian, 축소한 저차원 공간에서는 Student’s t 분포를 사용한다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195590203-082bd8b8-301c-41ee-b1ec-640bdb0a4f37.png" height="200"/>  
</p>

&nbsp;최적화과정은 기존의 방법론들과 거의 동일하며 다음과 cost function의 gradient에 조금 추가되는 부분이 생긴다.
<p align="center">
  <image src="https://user-images.githubusercontent.com/72682160/195590249-b9cf2161-3b2d-4665-ad77-62c203c9860d.png" height="200"/>  
</p>



## Reference
https://sustaining-starflower-aff.notion.site/2022-2-0e068bff3023401fa9fa13e96c0269d7  
(고려대학교 산업경영공학과 강필성 교수님 [IME654]Business-Analytics 수업자료)

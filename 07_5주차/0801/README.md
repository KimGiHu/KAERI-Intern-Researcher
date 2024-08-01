트랜스포머의 핵심 매커니즘 : 어텐션(Attention) 메커니즘 리뷰.

1. 트랜스포머 : Attentions is All you Need .. 어텐션 매커니즘의 등장 배경

1) 기존 Seq2Seq의 매커니즘
> 1. 인코더의 은닉 상태를 적절한값으로 초기화
> 2. 매 시점(time step)마다 원문의 단어(token)이 입력되면 인코더는 은닉상태를 업데이트.
> 3. 입력 시퀀스의 마지막까지 이 과정을 반복하면 최종 은닉상태가 담긴 축약된 정보인 고정된 문맥 벡터(context vector)가 생성됨.
> 4. context vector를 디코더에 넘겨주고, 디코더는 전달받은 context vector를 가지고서 은닉상태를 초기화함.
> 5. 매 시점(time step) 바로 직전 시점에 출력했던 단어를 입력받아 은닉 상태를 업데이트함.
> 6. <eos>토큰이 나올때까지의 1.~5. 과정을 반복해서 수행함.

2) 기존 Seq2Seq의 문제점
seq2seq의 대표적인 모델로 RNN, LSTM, GRU들이 있으며, 이 모델들은 이전 입력을 고려하여 다음 time step을 예상한다. 그러나, 짧은 reference window 크기를 가지고 있기에 입력이 길어지면 전체적인 시퀀스를 고려하지 못하게 되는 현상이 발생한다.
> * **병렬화문제** : 구조상 순차적으로 입력을 처리하기에, 대규모 데이터셋 같은 경우 학습시간이 길어졌다.
> * **Long Distance Dependency 문제** : reference window의 크기가 고정되어 있었기에, 입력 데이터가 길어지게 되면 떨어진 context vector들간의 관계성은 gradient vanishing/exploding 문제가 발생하게되어 학습이 원할지 않게 되었다.

2. 어텐션(Attention) 매커니즘

앞서 제시된 2가지의 주요 문제점을 해결하기 위해서, 새로운 학습 모델인 트랜스포머(Transforemr, 2017)모델이 제안되었다.
해당 모델의 주요 매커니즘인 어텐션(attention) 연산엥 대해 설명하도록 하겠다.


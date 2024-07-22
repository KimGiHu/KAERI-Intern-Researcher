우선, 실험을 함에 있어서 오류가 있음을 확인.

1. 데이터셋의 슬라이싱에 제대로 이루어지지 않아서, 정상 및 비정상 인코더 부분 학습이 RFQ 모듈에 대한 정보로만 이루어짐.

![Normal vs  Fault ver proposal](https://github.com/user-attachments/assets/fd7f5591-b819-4213-b407-1f95ee6b0a95)

2. 적응형 인코더 모델의 편향성으로 인해서, 결과가 잘 나오는 것으로 보여지나 실제로는 그렇지 아니한 모습을 보임.

![Normal vs  Fault ver proposal_loss](https://github.com/user-attachments/assets/79222ac5-bf05-4d76-bbf1-942a3499f1f6)

3. 해결방법 : 적응형 인코더를 학습하는 과정에서 새로운 아이디어가 필요한 것으로 보임 << 주요 contribution으로 작용할 듯.

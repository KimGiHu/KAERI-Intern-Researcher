Adaptive Learning 기법을 적용한 실험결과 정리
- - - - - - -
0. 기존 논문의 모델에 대한 unknown signal의 테스트 결과.

![Normal vs  Fault ver pretrain](https://github.com/user-attachments/assets/726b2a0c-67ad-4525-ae84-6917c25e4ebb)

- - - - - - -
1. Adaptive Model 학습에 사용한 정상신호와 사용하지 않은 비정상 신호 테스트. 
(정상신호 적응형 모델로 정상신호와 비정상신호를 테스트함.)

![Normal vs  Fault ver my proposal_adaptive_normal_unknown_fault](https://github.com/user-attachments/assets/c4881e6c-d0c6-4c69-87f9-5085f74a4530)

결과 정리 : 생각보다 잘 분리되지 않는 모습을 보임.


- - - - - - -
2. Adpative Model 학습시에 사용한 데이터로 테스트. 
(정상 적응형모델->정상신호, 비정상 적응형모델->비정상신호)

![Normal vs  Fault ver my proposal_best](https://github.com/user-attachments/assets/6bf07194-cb06-49f8-8c66-d6927b62f843)

결과 정리 : 제가 생각한 대로 normal과 abnormal신호가 미세하게 나마 분리되는 모습을 보입니다.


- - - - - - -
3. Adaptive Model + Ensemble Method 테스트. 
(**앙상블** 적응형모델-> 학습한 정상신호, **앙상블** 적응형모델-> 학습한 비정상신호)


![Normal vs  Fault ver my proposal_adapative_data](https://github.com/user-attachments/assets/a4eff6df-d25d-4ede-abfa-331211b0395d)

- - - - - - -
4. Adaptive Model로 unknown signal 테스트. 
(정상 적응형모델-> unknown 정상신호, 비정상 적응형모델-> unknown비정상신호)

![Normal vs  Fault ver my proposal_bestmodel_unknown_normal_fault](https://github.com/user-attachments/assets/194c3a7b-cd27-472d-b4f6-7b8945a2f7c7)


- - - - - - -
5. Adaptive Model + Ensemble Mehtod로 unknown signal 테스트

![Normal vs  Fault ver my proposal_ensemble_unknown_normal_fault](https://github.com/user-attachments/assets/ab29373a-59c2-4979-8e93-1c8926847904)



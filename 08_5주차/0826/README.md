### Mean and Standard deviation
Baseline Normal Loss - Mean:  0.015485567  Std:  0.003634747

Baselin Fault Loss - Mean:  0.026580421  Std:  0.018456949

Proposed Normal Loss - Mean:  0.0010912528  Std:  0.0005026587

Proposed Fault Loss - Mean:  0.020823535  Std:  0.027949354


### Baseline Model

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.017287295311689377

Precision: 0.7862

Recall: 0.5138

F1-score: 0.6214

Confusion Matrix:

[[1160  238]

 [ 828  875]]


![baseline Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/19061b81-f2c1-41b8-81ef-c94037a93f7c)


### Proposed Model

ROC Curve에서 정상신호와 비정산신호를 구분하는 최적의 값 : 0.0033925846219062805

Precision: 0.9947

Recall: 0.9994

F1-score: 0.9971

Confusion Matrix:

[[1389    9]

 [   1 1702]]


![proposed Normal vs  Fault ROC Curve](https://github.com/user-attachments/assets/6b7f0803-8791-492e-9b3a-20753477dad3)


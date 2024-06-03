같은 에러의 시계열 데이터들을 이어 붙여서 학습함.

ex) A-FLUX 신호에서 에러가 발생한 경우, A-FLUX 에러 신호들을 합해서 하나의 이상신호로 학습한다.

Xanomaly_append = min_max_scaler.fit_transform(Xanomaly[0,:,feature_index].reshape(-1,1))

for i in range(1,10):

    tmp = min_max_scaler.fit_transform(Xanomaly[i,:,feature_index].reshape(-1,1))
    
    Xanomaly_append = np.append(Xanomaly_append,tmp)


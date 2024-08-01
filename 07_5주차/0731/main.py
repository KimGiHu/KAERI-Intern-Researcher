import os 
import argparse

from torch.backends import cudnn
from utils import *

from mode import Solver

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True # pytorch에서 cuDNN 라이브러리의 동적방식 할당을 최적화함. >> 1. 변동하는 입력크기의 적절한 알고리즘 도입. 2. 컨볼루션 연산 최적화.
    # 학습한 모델의 저장경로 생성.
    if(not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config)) # vars : 모듈, 클래스, 클래스 인스턴스 객체(__dict__속성을 가지는 객체)에 대해 __dict__ (dictionary 형태)로 return해 주는 함수

    # config의 argument.mode가 'train'일 경우
    if config.mode == 'train':
        solver.train()
    # config의 argument.mode가 'test'일 경우
    elif config.mode == 'test':
        solver.test()

    return solver

if __name__ == '__main__':
    # 터미널 창에서 argument들 입력.
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4) # 학습률(Learning rate)
    parser.add_argument('--num_epochs', type=int, default=10) # 에포크 수(epoch)
    parser.add_argument('--k', type=int, default=3) # 
    parser.add_argument('--win_size', type=int, default=100) # 한 스텝당 학습및 예측에 쓰일 window 수 
    parser.add_argument('--input_c', type=int, default=38) # input data의 채널 수 
    parser.add_argument('--output_c', type=int, default=38) # output data의 채널 수 
    parser.add_argument('--batch_size', type=int, default=1024) # batch size 수
    parser.add_argument('--pretrained_model', type=str, default=None) # pretrained model 사용여부
    parser.add_argument('--dataset', type=str, default='credit') # 데이터셋 종류
    parser.add_argument('--mode', type=str, default='train', choices=['train','test']) # 학습 또는 테스트 모드 설정
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv') # 데이터셋의 저장경로 설정
    parser.add_argument('--model_save_path', type=str, default='checkpoints') # 모델의 저장경로 설정
    parser.add_argument('--anormly_ratio', type=float, default=4.00) # 이상치 비율. >> Association discrepancy를 계산할 때, 쓰임.

    config = parser.parse_args()

    args = vars(config) # config의 argument값들을 dictionary 형태로 return해줌.
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s : %s' % (str(k),str(v)))
    print('-------------- End ----------------')
    main(config)
    
    
    
        
    
    
    
    
    
    

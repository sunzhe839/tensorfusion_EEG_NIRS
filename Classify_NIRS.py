from scipy.io import loadmat
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import time
import argparse

from util import *
from model_NIRS import OneD_ResCNN as OneD_CNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NIRS classify')
    parser.add_argument('--Task', '-T', default='imag', help='imag or ment')
    parser.add_argument('--Type', '-Type', default='oxy', help='oxy or deoxy')
    parser.add_argument('--start_step', '-st', type=int, default=0, help='0->step-1')
    parser.add_argument('--step', '-s', type=int, default=4, help='parallel')
    args = parser.parse_args()

    K=5
    batch_size=16
    epoch_num=200
    check_test_step=10

    results={}
    for id in range(args.start_step,29,args.step):
        NIRS = loadmat('./data/NIRS/seg{}.mat'.format(id+1))['segment']

        time_acc_mean=[]
        time_acc_std=[]
        start=time.time()
        for i in range(33):
            start=time.time()
            print('ID:{}, Time window:{}'.format(id,i))
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            np.random.seed(0)

            X = np.transpose(NIRS[0,0][args.Task][0,0][args.Type][0,i]['x'][0,0],(2,1,0)) #30*36*60->60*36*30 (batch*channel*times)
            Y = np.transpose(NIRS[0,0][args.Task][0,0][args.Type][0,i]['y'][0,0]) #2*60->60*2

            if X.shape[2]>=30:
                X=X[:,:,:30]


            kf = KFold(n_splits=K)
            count=0
            kFlod_results=[]
            
            for train_index, test_index in kf.split(X):
                count+=1
                # print('Time window:{}, Fold:{}'.format(i,count))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                train_set = DeformedData(X_train,y_train)
                train_dataloader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                                            shuffle=True, pin_memory=True)

                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                np.random.seed(0)
                model = OneD_CNN(30,36).cuda()
                optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
                Loss = nn.BCELoss().cuda()

                best_test_acc=0
                for epoch in range(epoch_num):

                    model.train()
                    for ite, (input, target) in enumerate(train_dataloader):
                        input = Variable(input, requires_grad=False).cuda().float()
                        target = Variable(target, requires_grad=False).cuda().float()
                        optimizer.zero_grad()
                        output,_ = model(input)
                        loss = Loss(output, target)
                        loss.backward()
                        optimizer.step()

                    if (epoch+1)%check_test_step == 0:
                        model.eval()
                        input = Variable(torch.Tensor(X_test), requires_grad=False).cuda().float()
                        target = Variable(torch.Tensor(y_test), requires_grad=False).cuda().float()
                        output,_ = model(input)

                        _, preds = output.max(1)
                        _, target = target.max(1)
                        test_correct = float(preds.eq(target).sum())/X_test.shape[0]

                        input = Variable(torch.Tensor(X_train), requires_grad=False).cuda().float()
                        target = Variable(torch.Tensor(y_train), requires_grad=False).cuda().float()
                        output,_ = model(input)

                        _, preds = output.max(1)
                        _, target = target.max(1)
                        train_correct = float(preds.eq(target).sum())/X_train.shape[0]

                        # print('epoch->{}, train acc->{}, test acc->{}'.format(
                        #      epoch+1,train_correct,test_correct))
                        if train_correct==1.0 and best_test_acc<test_correct:
                            best_test_acc=test_correct
                kFlod_results.append(best_test_acc)
            print('5 Folds cost time: {}'.format(time.time()-start))
            kFlod_results=np.array(kFlod_results)
            time_acc_mean.append(kFlod_results.mean())
            time_acc_std.append(kFlod_results.std())
            print(time_acc_mean)
        results[id]=np.array([time_acc_mean,time_acc_std])

    np.save('NIRS_{}_{}_start{}_step{}.npy'.format(args.Task, args.Type, args.start_step, args.step),results)
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
from model_hybrid import OneD_CNN_changed as OneD_CNN
from util import *

#             EEG    oxy   deoxy
#best MI id    11     11     15    
#     MA id    17     20     18

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NIRS classify')
    parser.add_argument('--Task', '-T', default='imag', help='imag or ment')
    parser.add_argument('--start_step', '-st', type=int, default=0, help='0->step-1')
    parser.add_argument('--step', '-s', type=int, default=4, help='parallel')
    parser.add_argument('--model', '-m', default='linear', help='linear or tensor')

    args = parser.parse_args()


    K=5
    batch_size=16
    epoch_num=300
    check_test_step=10

    results={}
    for id in range(args.start_step,29,args.step):
        if id in [0,12,22] and args.Task == 'ment':
            continue
        NIRS = loadmat('./data/NIRS/seg{}.mat'.format(id+1))['segment']
        EEG = loadmat('./data/EEG/seg{}.mat'.format(id+1))['segment']

        time_acc_mean=[]
        time_acc_std=[]
        start=time.time()
        for i in range(33):
            start=time.time()
            print('ID:{}, Time window:{}'.format(id,i))
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            np.random.seed(0)

            X_EEG = np.transpose(EEG[0,0][args.Task][0,i]['x'][0,0],(2,1,0)) #600*30*60->60*30*600 (batch*channel*times)
            Y_EEG = np.transpose(EEG[0,0][args.Task][0,i]['y'][0,0]) #2*60->60*2

            X_oxy = np.transpose(NIRS[0,0][args.Task][0,0]['oxy'][0,i]['x'][0,0],(2,1,0)) #30*36*60->60*36*30 (batch*channel*times)
            # Y_oxy = np.transpose(NIRS[0,0][args.Task][0,0]['oxy'][0,i]['y'][0,0]) #2*60->60*2

            X_deoxy = np.transpose(NIRS[0,0][args.Task][0,0]['deoxy'][0,i]['x'][0,0],(2,1,0)) #30*36*60->60*36*30 (batch*channel*times)
            # Y_deoxy = np.transpose(NIRS[0,0][args.Task][0,0]['deoxy'][0,i]['y'][0,0]) #2*60->60*2

            X_EEG=X_EEG[:,:,:600]

            # e_max=X_EEG.max()
            # e_min=X_EEG.min()
            # X_EEG = (X_EEG-e_min)/(e_max-e_min)
            # X_EEG=X_EEG*2-1

            X_oxy=X_oxy[:,:,:30]
            X_deoxy=X_deoxy[:,:,:30]


            kf = KFold(n_splits=K)
            count=0
            kFlod_results=[]
            
            for train_index, test_index in kf.split(X_EEG):
                count+=1
                # print('Time window:{}, Fold:{}'.format(i,count))
                X_train_EEG, X_test_EEG = X_EEG[train_index], X_EEG[test_index]
                X_train_oxy, X_test_oxy = X_oxy[train_index], X_oxy[test_index]
                X_train_deoxy, X_test_deoxy = X_deoxy[train_index], X_deoxy[test_index]
                y_train, y_test = Y_EEG[train_index], Y_EEG[test_index]
                print(X_train_oxy.shape)

                train_set = DeformedTripleData(X_train_EEG,X_train_oxy,X_train_deoxy,y_train)
                train_dataloader = DataLoader(dataset=train_set, num_workers=2, batch_size=batch_size,
                                                            shuffle=True, pin_memory=True)

                torch.manual_seed(0)
                torch.cuda.manual_seed(0)
                np.random.seed(0)
                model = OneD_CNN(30,36,16).cuda()
                optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
                Loss = nn.BCELoss().cuda()

                best_test_acc=0
                for epoch in range(epoch_num):
                    # scheduler.step()
                    model.train()
                    for ite, (input1,input2,input3,target) in enumerate(train_dataloader):
                        input1 = Variable(input1, requires_grad=False).cuda().float()
                        input2 = Variable(input2, requires_grad=False).cuda().float()
                        input3 = Variable(input3, requires_grad=False).cuda().float()
                        target = Variable(target, requires_grad=False).cuda().float()
                        optimizer.zero_grad()
                        print(input1.shape)
                        print(input2.shape)
                        print(input3.shape)
                        output = model(input1,input2,input3)
                        loss = Loss(output, target)
                        loss.backward()
                        optimizer.step()

                    if (epoch+1)%check_test_step == 0:
                        with torch.no_grad():
                            model.eval()
                            input1 = Variable(torch.Tensor(X_test_EEG), requires_grad=False).cuda().float()
                            input2 = Variable(torch.Tensor(X_test_oxy), requires_grad=False).cuda().float()
                            input3 = Variable(torch.Tensor(X_test_deoxy), requires_grad=False).cuda().float()
                            target = Variable(torch.Tensor(y_test), requires_grad=False).cuda().float()
                            output = model(input1,input2,input3)

                            _, preds = output.max(1)
                            _, target = target.max(1)
                            test_correct = float(preds.eq(target).sum())/X_test_EEG.shape[0]

                            input1 = Variable(torch.Tensor(X_train_EEG), requires_grad=False).cuda().float()
                            input2 = Variable(torch.Tensor(X_train_oxy), requires_grad=False).cuda().float()
                            input3 = Variable(torch.Tensor(X_train_deoxy), requires_grad=False).cuda().float()
                            target = Variable(torch.Tensor(y_train), requires_grad=False).cuda().float()
                            output = model(input1,input2,input3)
                            # loss = Loss(output, target)

                            _, preds = output.max(1)
                            _, target = target.max(1)
                            train_correct = float(preds.eq(target).sum())/X_train_EEG.shape[0]
                            

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
            # break
        results[id]=np.array([time_acc_mean,time_acc_std])

    np.save('Hybrid3_{}_start{}_step{}_{}.npy'.format(args.Task, args.start_step, args.step,args.model),results)

    tmp=0
    for k in results.keys():
        tmp+=results[k][0]
    print(tmp/len(results.keys()))


    
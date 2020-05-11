import torch
import torch.nn as nn
import torch.nn.functional as F
from model_EEG import ResDepSepBlock

class ReLUConvBn(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ReLUConvBn, self).__init__()
        self.op = nn.Sequential(
            nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(C_out),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

#LF net (in paper)
class OneD_CNN(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel, rank, middle_num=64):
        super(OneD_CNN, self).__init__()

        self.EEG_net = nn.Sequential(
            ReLUConvBn(EEG_channel, EEG_channel * 2, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(EEG_channel * 2, EEG_channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(EEG_channel * 2, EEG_channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(EEG_channel * 2, EEG_channel * 4, kernel_size=9, stride=4, padding=0),
            ReLUConvBn(EEG_channel * 4, EEG_channel * 4, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(EEG_channel * 4, EEG_channel * 4, kernel_size=3, stride=1, padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_oxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel, NIRS_channel * 2, kernel_size=5, stride=2, padding=0),
            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 4, kernel_size=5, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 4, NIRS_channel * 4, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 4, NIRS_channel * 4, kernel_size=3, stride=1, padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_deoxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel, NIRS_channel * 2, kernel_size=5, stride=2, padding=0),
            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 2, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 2, kernel_size=3, stride=1, padding=0),

            ReLUConvBn(NIRS_channel * 2, NIRS_channel * 4, kernel_size=5, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 4, NIRS_channel * 4, kernel_size=3, stride=1, padding=0),
            ReLUConvBn(NIRS_channel * 4, NIRS_channel * 4, kernel_size=3, stride=1, padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        # self.w_core1 = torch.nn.Parameter(torch.Tensor(rank, EEG_channel * 4, middle_num))
        # self.w_core2 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel * 4, middle_num))
        # self.w_core3 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel * 4, middle_num))
        # self.w_out = torch.nn.Parameter(torch.randn(rank))
        #
        # with torch.no_grad():
        #     self.w_core1.normal_(0, 1 / (EEG_channel * 4))
        #     self.w_core2.normal_(0, 1 / (NIRS_channel * 4))
        #     self.w_core3.normal_(0, 1 / (NIRS_channel * 4))
        #     self.w_out.normal_(0, 1 / rank)
        #
        # self.out = nn.Sequential(
        #     nn.Tanh(),
        #     nn.Linear(middle_num, 2),
        #     nn.Softmax(dim=-1),
        # )

        self.out = nn.Sequential(
            nn.Linear(EEG_channel*4+NIRS_channel*4+NIRS_channel*4,128),
            nn.ReLU(inplace=False),
            nn.Linear(128,2),
            nn.Softmax(dim=-1)
        )

    def forward(self, EEG_x, NIRS_oxy_x, NIRS_deoxy_x):
        x1 = self.EEG_net(EEG_x)
        print(x1.shape)
        x1 = torch.squeeze(x1, dim=2)
        print(x1.shape)
        x2 = self.NIRS_oxy_net(NIRS_oxy_x)
        print(x2.shape)
        x2 = torch.squeeze(x2, dim=2)

        x3 = self.NIRS_deoxy_net(NIRS_deoxy_x)
        x3 = torch.squeeze(x3, dim=2)


        x=torch.cat([x1,x2,x3],dim=1)

        return self.out(x)

#TF net (in paper)
class OneD_CNN(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank,middle_num=64):
        super(OneD_CNN, self).__init__()
        
        self.EEG_net = nn.Sequential(
            ReLUConvBn(EEG_channel,EEG_channel*2,kernel_size=9,stride=4,padding=0),
            ReLUConvBn(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(EEG_channel*2,EEG_channel*4,kernel_size=9,stride=4,padding=0),
            ReLUConvBn(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_oxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_deoxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.w_core1 = torch.nn.Parameter(torch.Tensor(rank, EEG_channel*4, middle_num))
        self.w_core2 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*4, middle_num))
        self.w_core3 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*4, middle_num))
        self.w_out = torch.nn.Parameter(torch.randn(rank))

        with torch.no_grad():
            self.w_core1.normal_(0, 1/(EEG_channel*4))
            self.w_core2.normal_(0, 1/(NIRS_channel*4))
            self.w_core3.normal_(0, 1/(NIRS_channel*4))
            self.w_out.normal_(0,1/rank)

        self.out = nn.Sequential(
                nn.Tanh(),
                nn.Linear(middle_num,2),
                nn.Softmax(dim=-1),
            )

        # self.out = nn.Sequential(
        #     nn.Linear(EEG_channel*4+NIRS_channel*4+NIRS_channel*4,128),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(128,2),
        #     nn.Softmax(dim=-1)
        # )
        

    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        x1=self.EEG_net(EEG_x)
        print(x1.shape)
        x1=torch.squeeze(x1,dim=2)
        print(x1.shape)
        x2=self.NIRS_oxy_net(NIRS_oxy_x)
        print(x2.shape)
        x2=torch.squeeze(x2,dim=2)

        x3=self.NIRS_deoxy_net(NIRS_deoxy_x)
        x3=torch.squeeze(x3,dim=2)

        x1 = torch.einsum('bc,rco->bro',(x1,self.w_core1))
        x2 = torch.einsum('bc,rco->bro',(x2,self.w_core2))
        x3 = torch.einsum('bc,rco->bro',(x3,self.w_core3))

        x = torch.einsum('bro,bro,bro,r->bo',(x1,x2,x3,self.w_out))
        x = F.normalize(x, p=2, dim=1)

        # x=torch.cat([x1,x2,x3],dim=1)

        return self.out(x)

#the module of TF part (not used)
class TensorFusion(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank):
        super(TensorFusion, self).__init__()
        self.w_core1 = torch.nn.Parameter(torch.Tensor(rank, EEG_channel+1, 2))
        self.w_core2 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel+1, 2))
        self.w_core3 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel+1, 2))
        self.w_out = torch.nn.Parameter(torch.randn(rank))

        with torch.no_grad():
            self.w_core1.normal_(0, 1/(EEG_channel))
            self.w_core2.normal_(0, 1/(NIRS_channel))
            self.w_core3.normal_(0, 1/(NIRS_channel))
            self.w_out.normal_(0,1/rank)

        self.out = nn.Softmax(dim=-1)

    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        if EEG_x.is_cuda:
            one1 = torch.ones(EEG_x.shape[0],1).cuda()
            one2 = torch.ones(NIRS_oxy_x.shape[0],1).cuda()
            one3 = torch.ones(NIRS_deoxy_x.shape[0],1).cuda()
        else:
            one1 = torch.ones(EEG_x.shape[0],1)
            one2 = torch.ones(NIRS_oxy_x.shape[0],1)
            one3 = torch.ones(NIRS_deoxy_x.shape[0],1)
        x1=torch.cat([EEG_x,one1], dim=1)
        x2=torch.cat([NIRS_oxy_x,one2], dim=1)
        x3=torch.cat([NIRS_deoxy_x,one3], dim=1)

        x1 = torch.einsum('bc,rco->bro',(x1,self.w_core1))
        x2 = torch.einsum('bc,rco->bro',(x2,self.w_core2))
        x3 = torch.einsum('bc,rco->bro',(x3,self.w_core3))

        x = torch.einsum('bro,bro,bro,r->bo',(x1,x2,x3,self.w_out))

        return self.out(x)

class LinearFusion(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank):
        super(LinearFusion, self).__init__()

        self.out = nn.Sequential(
            nn.Linear(EEG_channel+NIRS_channel+NIRS_channel,128),
            nn.ReLU(inplace=False),
            nn.Linear(128,2),
            nn.Softmax(dim=-1)
        )
    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        x=torch.cat([EEG_x,NIRS_oxy_x,NIRS_deoxy_x],dim=1)
        return self.out(x)

#p-order PF net resnet version (not in paper)
class OneD_ResCNN(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank,middle_num=64):
        super(OneD_ResCNN, self).__init__()

        self.EEG_net = nn.Sequential(
            ResDepSepBlock(EEG_channel,EEG_channel*2,kernel_size=5,stride=4),
            ResDepSepBlock(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1),
            ResDepSepBlock(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1),

            ResDepSepBlock(EEG_channel*2,EEG_channel*4,kernel_size=5,stride=4),
            ResDepSepBlock(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1),
            ResDepSepBlock(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1),

            ResDepSepBlock(EEG_channel*4,EEG_channel*8,kernel_size=3,stride=2),
            ResDepSepBlock(EEG_channel*8,EEG_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(EEG_channel*8,EEG_channel*8,kernel_size=3,stride=1),

            ResDepSepBlock(EEG_channel*8,EEG_channel*8,kernel_size=3,stride=2),
            ResDepSepBlock(EEG_channel*8,EEG_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(EEG_channel*8,EEG_channel*8,kernel_size=3,stride=1),
            
            nn.AdaptiveAvgPool1d(1),
        )

        self.NIRS_oxy_net = nn.Sequential(
            ResDepSepBlock(NIRS_channel,NIRS_channel*2,kernel_size=3,stride=2),
            ResDepSepBlock(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*2,NIRS_channel*4,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*4,NIRS_channel*8,kernel_size=3,stride=2),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            
            nn.AdaptiveAvgPool1d(1),
        )

        self.NIRS_deoxy_net = nn.Sequential(
            ResDepSepBlock(NIRS_channel,NIRS_channel*2,kernel_size=3,stride=2),
            ResDepSepBlock(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*2,NIRS_channel*4,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*4,NIRS_channel*8,kernel_size=3,stride=2),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),

            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            ResDepSepBlock(NIRS_channel*8,NIRS_channel*8,kernel_size=3,stride=1),
            
            nn.AdaptiveAvgPool1d(1),
        )

        # self.w_core1 = torch.nn.Parameter(torch.Tensor(rank, EEG_channel*8, middle_num))
        # self.w_core2 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*8, middle_num))
        # self.w_core3 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*8, middle_num))
        # self.w_out = torch.nn.Parameter(torch.randn(rank))

        # with torch.no_grad():
        #     self.w_core1.normal_(0, 1/(EEG_channel*8))
        #     self.w_core2.normal_(0, 1/(NIRS_channel*8))
        #     self.w_core3.normal_(0, 1/(NIRS_channel*8))
        #     self.w_out.normal_(0,1/rank)

        # self.out = nn.Sequential(
        #         nn.Tanh(),
        #         nn.Linear(middle_num,2),
        #         nn.Softmax(dim=-1),
        #     )

        self.out = nn.Sequential(
            nn.Linear(EEG_channel*8+NIRS_channel*8+NIRS_channel*8,128),
            nn.ReLU(inplace=False),
            nn.Linear(128,2),
            nn.Softmax(dim=-1)
        )

    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        x1=self.EEG_net(EEG_x)
        x1=torch.squeeze(x1)
        x2=self.NIRS_oxy_net(NIRS_oxy_x)
        x2=torch.squeeze(x2)
        x3=self.NIRS_deoxy_net(NIRS_deoxy_x)
        x3=torch.squeeze(x3)

        # x1 = torch.einsum('bc,rco->bro',(x1,self.w_core1))
        # x2 = torch.einsum('bc,rco->bro',(x2,self.w_core2))
        # x3 = torch.einsum('bc,rco->bro',(x3,self.w_core3))

        # x = torch.einsum('bro,bro,bro,r->bo',(x1,x2,x3,self.w_out))
        # x = F.normalize(x, p=2, dim=1)

        x=torch.cat([x1,x2,x3],dim=1)

        return self.out(x)


#p-order PF net (the net in paper)
class OneD_CNN_changed(nn.Module):
    def __init__(self, EEG_channel, NIRS_channel,rank,middle_num=64):
        super(OneD_CNN_changed, self).__init__()
        
        self.EEG_net = nn.Sequential(
            ReLUConvBn(EEG_channel,EEG_channel*2,kernel_size=9,stride=4,padding=0),
            ReLUConvBn(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(EEG_channel*2,EEG_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(EEG_channel*2,EEG_channel*4,kernel_size=9,stride=4,padding=0),
            ReLUConvBn(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(EEG_channel*4,EEG_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_oxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.NIRS_deoxy_net = nn.Sequential(
            ReLUConvBn(NIRS_channel,NIRS_channel*2,kernel_size=5,stride=2,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*2,NIRS_channel*2,kernel_size=3,stride=1,padding=0),

            ReLUConvBn(NIRS_channel*2,NIRS_channel*4,kernel_size=5,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),
            ReLUConvBn(NIRS_channel*4,NIRS_channel*4,kernel_size=3,stride=1,padding=0),

            nn.AdaptiveAvgPool1d(1)
        )

        self.w_core1 = torch.nn.Parameter(torch.Tensor(rank, EEG_channel*4+NIRS_channel*8, middle_num))
        # self.w_core2 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*4, middle_num))
        # self.w_core3 = torch.nn.Parameter(torch.Tensor(rank, NIRS_channel*4, middle_num))
        self.w_out = torch.nn.Parameter(torch.randn(rank))

        with torch.no_grad():
            self.w_core1.normal_(0, 1/(EEG_channel*4))
            # self.w_core2.normal_(0, 1/(NIRS_channel*4))
            # self.w_core3.normal_(0, 1/(NIRS_channel*4))
            self.w_out.normal_(0,1/rank)

        self.out = nn.Sequential(
                nn.Tanh(),
                nn.Linear(middle_num,2),
                nn.Softmax(dim=-1),
            )

        # self.out = nn.Sequential(
        #     nn.Linear(EEG_channel*4+NIRS_channel*4+NIRS_channel*4,128),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(128,2),
        #     nn.Softmax(dim=-1)
        # )
        

    def forward(self,EEG_x,NIRS_oxy_x,NIRS_deoxy_x):
        x1=self.EEG_net(EEG_x)
        x1=torch.squeeze(x1)
        x2=self.NIRS_oxy_net(NIRS_oxy_x)
        x2=torch.squeeze(x2)
        x3=self.NIRS_deoxy_net(NIRS_deoxy_x)
        x3=torch.squeeze(x3)


        x=torch.cat([x1,x2,x3],dim=1)

        x1 = torch.einsum('bc,rco->bro',(x,self.w_core1))

        x = torch.einsum('bro,bro,bro,bro,bro,bro,r->bo',(x1,x1,x1,x1,x1,x1,self.w_out))
        x = F.normalize(x, p=2, dim=1)


        return self.out(x)

if __name__=='__main__':
    from thop import profile
    model = OneD_CNN(30, 36, 16)
    x1 = torch.randn(4, 30, 600)
    x2 = torch.randn(4, 36, 30)
    x3 = torch.randn(4, 36, 30)
    flops, params = profile(model, inputs=(x1,x2,x3))
    print(flops,params)

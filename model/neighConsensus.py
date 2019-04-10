import torch
from conv4D import Conv4d
import torch.nn.functional as F

class NeighConsensus(torch.nn.Module):
    def __init__(self, kernel_sizes=[5,5,5], channels=[16,16,1]):
        super(NeighConsensus, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        self.model = []
        channels = [1] + channels ## the initial input channel is 1
        for i in range(num_layers):
            ch_in = channels[i] 
            ch_out = channels[i + 1]
            k_size = kernel_sizes[i]
            self.model += [Conv4d(in_channels=ch_in,out_channels=ch_out,kernel_size=k_size,bias=True), 
                               torch.nn.ReLU(inplace=True)]
        self.model = torch.nn.Sequential(*self.model)        
        

    def forward(self, x):
        # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        corr4d = self.model(x)+self.model(x.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
        # because of the ReLU layers in between linear layers, 
        # this operation is different than convolving a single time with the filters+filters^T
        # and therefore it makes sense to do this.
        return corr4d

def MutualMatching(corr4d):
    # mutual matching
    b,_,h,w,d,t = corr4d.size()

    corr4d_B=corr4d.view(b, h * w, d, t) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(b, h, w, d * t)

    # get max
    corr4d_B_max,_=torch.max(corr4d_B,dim=1,keepdim=True)
    corr4d_A_max,_=torch.max(corr4d_A,dim=3,keepdim=True)

    eps = 1e-5
    corr4d_B=corr4d_B/(corr4d_B_max+eps)
    corr4d_A=corr4d_A/(corr4d_A_max+eps)

    corr4d_B=corr4d_B.view(corr4d.size())
    corr4d_A=corr4d_A.view(corr4d.size())

    corr4d=corr4d*(corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
        
    return corr4d
    
def MutualMatchingSoftMax(corr4d):
    # mutual matching
    b,_,h,w,d,t = corr4d.size()

    corr4d_B=corr4d.view(b, h * w, d, t) # [batch_idx,k_A,i_B,j_B]
    corr4d_A=corr4d.view(b, h, w, d * t)

    # get max
    corr4d_B_softmax = F.softmax(corr4d_B, dim = 1)
    corr4d_A_softmax =F.softmax(corr4d_A, dim = 3)
    
    
    corr4d_B= corr4d_B_softmax.view(corr4d.size())
    corr4d_A= corr4d_A_softmax.view(corr4d.size())

    corr4d=corr4d * (corr4d_A*corr4d_B) # parenthesis are important for symmetric output 
    return corr4d

    
if __name__ == '__main__' : 
    print ('Test NeighConsensus Network...')
    model = NeighConsensus()
    a = torch.randn(1, 1, 11, 12, 13 ,14).cuda()
    model.cuda()
    print (model(a).size()) # output is 1, 1, 11, 12, 13, 14
    
    
    print ('Test Original Mutual Matching....')
    corr4d = MutualMatching(a)
    print (corr4d.size()) # output is 1, 1, 11, 12, 13, 14
    
    print ('Test SoftMax Mutual Matching....')
    corr4d = MutualMatchingSoftMax(a)
    print (corr4d.size()) # output is 1, 1, 11, 12, 13, 14
    
    
    

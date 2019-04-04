import torch


def featureL2Norm(feature):
    eps = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+eps,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)
    
def featureCorrelation(feature_A, feature_B) : 
    
    b,c,hA,wA = feature_A.size()
    b,c,hB,wB = feature_B.size()
    # reshape features for matrix multiplication
    feature_A = feature_A.view(b,c,hA*wA).transpose(1,2) # size [b,c,h*w]
    feature_B = feature_B.view(b,c,hB*wB) # size [b,c,h*w]
    # perform matrix mult.
    feature_mul = torch.bmm(feature_A,feature_B)
    # indexed [batch,row_A,col_A,row_B,col_B]
    corr4d = feature_mul.view(b,hA,wA,hB,wB).unsqueeze(1) ## b, c, h, w, d, t
    
    return corr4d
        
if __name__ == '__main__' : 
    print ('Test feature L2 Normalization...')
    feature = torch.randn(1, 256, 16 ,16).cuda()
    featureNorm = featureL2Norm(feature)
    print (torch.sum(featureNorm ** 2, dim=1))
    
    print ('Test featureCorrelation...')
    featA = torch.randn(1, 256, 16 ,16).cuda()
    featB = torch.randn(1, 256, 16 ,16).cuda()
    featANorm = featureL2Norm(featA)
    featBNorm = featureL2Norm(featB)
    corr4d = featureCorrelation(featANorm, featBNorm)
    
    print (corr4d.size())
    


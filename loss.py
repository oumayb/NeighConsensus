import torch
import numpy as np

def weakLossBatch(model, batch, softmaxMM):

    normalize = lambda x : x if softmaxMM else lambda x: torch.nn.functional.softmax(x,1) 
    corr4d = model(batch['source_image'], batch['target_image'])
    
    b, w = corr4d.size()[0], corr4d.size()[2]
    
    corr4dA =corr4d.view(b, w * w, w, w) 
    corr4dB =corr4d.view(b, w, w, w * w).permute(0,3,1,2) 
    
    normA = normalize(corr4dA)
    normB = normalize(corr4dB)
    
    # compute matching scores    
    scoreB,_= torch.max(normB,dim=1)
    scoreA,_= torch.max(scoreA,dim=1)
    score = torch.mean(scoreA + scoreB)/2
    
    return score
    
def WeakLoss(model, batch, softmaxMM) : 
    scorePos = weakLossBatch(model, batch, softmaxMM)
    b = batch['source_image'].size()[0]
    batch['source_image']=batch['source_image'][np.roll(np.arange(b),-1),:]
    scoreNeg = weakLossBatch(model, batch, softmaxMM)
    loss = scoreNeg - scorePos
    return loss
    



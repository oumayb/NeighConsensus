import torch 
from neighConsensus import NeighConsensus, MutualMatching, MutualMatchingSoftMax
from util import featureL2Norm, featureCorrelation
from FeatureExtractor.ResNet18 import ResNetConv4, ResNetConv5

class NCNet(torch.nn.Module):
    def __init__(self, kernel_sizes, channels, featExtractor, featExtractorPth, finetuneFeatExtractor, softmaxMutualMatching):
        super(NCNet, self).__init__()
        
        ## Define feature extractor
        if featExtractor == 'ResNet18Conv4' : 
            self.featExtractor = ResNetConv4(featExtractorPth) 
        elif featExtractor == 'ResNet18Conv5' :
            self.featExtractor = ResNetConv5(featExtractorPth) 
            
         if finetuneFeatExtractor : 
             msg = 'Set Feature Extraction validation mode...'
             print (msg)
             self.featExtractor.eval()

         ## Mutual Matching method
        self.mutualMatch = MutualMatchingSoftMax if softmaxMutualMatching else MutualMatching
        
        ## NeighConsensus
        self.NeighConsensus = NeighConsensus(kernel_sizes, channels)
        
        

    def forward(self, xA, xB):
        ## Extract Feature
        featA, featB = self.featExtractor(xA), self.featExtractor(xA)
        
        ## Normalization
        featANorm, featBNorm = featureL2Norm(featA), featureL2Norm(featB)
        
        ## Correlation Tensor
        corr4d = featureCorrelation(featANorm, featBNorm)
        
        ## Neighbor Consensus
        corr4d = self.NeighConsensus(corr4d)
        
        ## Mutual Match 
        corr4d = self.mutualMatch(corr4d)
        
        return corr4d
        
        
if __name__ == '__main__' : 
    print ('Test NcNet...')
    model = NCNet(kernel_sizes=[5,5,5], channels=[16,16,1], featExtractor = 'ResNet18Conv4', featExtractorPth = './model/FeatureExtractor/resnet18.pth', softmaxMutualMatching = True)
    a = torch.randn(1, 3, 224, 224).cuda()
    b = torch.randn(1, 3, 224, 224).cuda()
    
    model.cuda()
    
    print (model(a, b).size()) # output is 1, 1, 11, 12, 13, 14
   

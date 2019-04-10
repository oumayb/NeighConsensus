import torch 
from neighConsensus import NeighConsensus, MutualMatching, MutualMatchingSoftMax
from util import featureL2Norm, featureCorrelation
from FeatureExtractor.ResNet18 import ResNetConv4, ResNetConv5
import torchvision.models as models
from torch import nn

class NCNet(torch.nn.Module):
    def __init__(self, kernel_sizes, channels, featExtractor, featExtractorPth, finetuneFeatExtractor, softmaxMutualMatching):
        super(NCNet, self).__init__()
        
        ## Define feature extractor
        if featExtractor == 'ResNet18Conv4' : 
            self.featExtractor = ResNetConv4(featExtractorPth) 
        elif featExtractor == 'ResNet18Conv5' :
            self.featExtractor = ResNetConv5(featExtractorPth)
        elif featExtractor=='ResNet101Conv4':
            print ('Loading ResNet 101 Conv4 Weight ...')
            self.featExtractor = models.resnet101(pretrained=True)            
            resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']
            resnet_module_list = [getattr(self.featExtractor,l) for l in resnet_feature_layers]
            last_layer_idx = resnet_feature_layers.index('layer3')
            self.featExtractor = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

            
        if not finetuneFeatExtractor : 
            msg = '\nSet Feature Extractor to validation mode...'
            print (msg)
            self.featExtractor.eval()

         ## Mutual Matching method
        self.mutualMatch = MutualMatchingSoftMax if softmaxMutualMatching else MutualMatching
        
        ## NeighConsensus
        self.NeighConsensus = NeighConsensus(kernel_sizes, channels)
        
        

    def forward(self, xA, xB):
        ## Extract Feature
        featA, featB = self.featExtractor(xA), self.featExtractor(xB)
        ## Normalization
        featANorm, featBNorm = featureL2Norm(featA), featureL2Norm(featB)
        #print ('feat L2:', featANorm[0, 0, :, :])
        ## Correlation Tensor
        corr4d = featureCorrelation(featANorm, featBNorm)
        #print ('corr 4d feature:', corr4d[0, 0, 0, 0, :, :])
        ## Mutual Match 
        corr4d = self.mutualMatch(corr4d)
        #print ('mutual match:', corr4d[0, 0, 0, 0, :, :])
        ## Neighbor Consensus
        corr4d = self.NeighConsensus(corr4d)
        #print ('output neigh consensus:', corr4d[0, 0, 0, 0, :, :])
        ## Mutual Match 
        corr4d = self.mutualMatch(corr4d)
        #print ('final mutual match:', corr4d[0, 0, 0, 0, :, :])
        return corr4d
        
        
if __name__ == '__main__' : 
    print ('Test NcNet...')
    model = NCNet(kernel_sizes=[5,5,5], channels=[16,16,1], featExtractor = 'ResNet18Conv4', featExtractorPth = './model/FeatureExtractor/resnet18.pth', softmaxMutualMatching = True)
    a = torch.randn(1, 3, 224, 224).cuda()
    b = torch.randn(1, 3, 224, 224).cuda()
    
    model.cuda()
    
    print (model(a, b).size()) # output is 1, 1, 11, 12, 13, 14
   

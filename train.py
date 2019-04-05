import sys 
sys.path.append('./model')

import argparse 
import torch
import numpy as np
from model.model import NCNet
import torchvision.transforms as transforms
from dataloader import TrainLoader, ValLoader
from loss import WeakLoss
import torch.optim as optim
import json 
import os 

## Parameters
parser = argparse.ArgumentParser(description='Nc-Net Training')

## Input / Output 
parser.add_argument('--outDir', type=str, help='output model directory')
parser.add_argument('--resumePth', type=str, help='resume model path')
parser.add_argument('--featExtractorPth', type=str, default = 'model/FeatureExtractor/resnet18.pth', help='feature extractor path')
parser.add_argument('--imgDir', type=str, default = 'data/pf-pascal/JPEGImages/', help='image Directory')
parser.add_argument('--trainCSV', type=str, default = 'data/pf-pascal/train.csv', help='train csv')
parser.add_argument('--valCSV', type=str, default = 'data/pf-pascal/val.csv', help='val csv')
parser.add_argument('--imgSize', type=int, default = 400, help='train image size')


## learning parameter
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batchSize', type=int, default=16, help='batch size')
parser.add_argument('--nbEpoch', type=int, default=5, help='number of training epochs')
parser.add_argument('--neighConsKernel', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--neighConsChannel', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
parser.add_argument('--finetuneFeatExtractor', action='store_true', help='whether fine-tuning feature extractor')
parser.add_argument('--featExtractor', type=str, default='ResNet18Conv4', choices=['ResNet18Conv4', 'ResNet18Conv5'], help='feature extractor')
parser.add_argument('--cuda', action='store_true', help='GPU setting')
parser.add_argument('--softmaxMM', action='store_true', help='whether use softmax Mutual Matching')

args = parser.parse_args()
print(args)


## Set seed 
torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

## Initial Model
model = NCNet(kernel_sizes=args.neighConsKernel, 
              channels=args.neighConsChannel, 
              featExtractor = args.featExtractor, 
              featExtractorPth = args.featExtractorPth, 
              finetuneFeatExtractor = args.finetuneFeatExtractor,
              softmaxMutualMatching = args.softmaxMM)

if not args.finetuneFeatExtractor:
    msg = 'Ignore the gradient for the parameters in the feature extractor'
    print (msg)
    for p in model.featExtractor.parameters(): 
        p.requires_grad=False
        


if args.resumePth : 
    msg = '\nResume from {}'.format(args.resumePth)
    model.load_state_dict(torch.load(args.resumePth))
    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

## Train Val DataLoader
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
trainTransform = transforms.Compose([transforms.RandomResizedCrop(args.imgSize),
                                     transforms.ToTensor(), 
                                     normalize,])

valTransform = transforms.Compose([transforms.Resize(args.imgSize), 
                                     transforms.CenterCrop(args.imgSize), 
                                     transforms.ToTensor(), 
                                     normalize,])

trainLoader = TrainLoader(batchSize=args.batchSize, 
                          pairCSV=args.trainCSV, 
                          imgDir = args.imgDir, 
                          trainTransform = trainTransform)
                          
valLoader = ValLoader(batchSize=args.batchSize, 
                          pairCSV=args.valCSV, 
                          imgDir = args.imgDir, 
                          valTransform = valTransform)


if not os.path.exists(args.outDir) : 
    os.mkdir(args.outDir)
    
    
# Train
bestValLoss = np.inf
history = {'TrainLoss' : [], 'ValLoss' : []}
outHistory = os.path.join(args.outDir, 'history.json')
outModel = os.path.join(args.outDir, 'netBest.pth')

for epoch in range(1, args.nbEpoch + 1) : 
    trainLoss = 0.
    valLoss = 0.
    for i, batch in enumerate(trainLoader) : 
        
        optimizer.zero_grad()
        
        loss = WeakLoss(model, batch, args.softmaxMM)
        loss.backward()
        
        optimizer.step()
        
        trainLoss += loss.item()
        if i % 100 == 99 : 
            msg = 'Epoch {:d}, Batch {:d}, Train Loss : {:.4f}'.format(epoch, i + 1, trainLoss / (i + 1))
            print (msg)
            
    ## Validation 
    trainLoss = trainLoss / len(trainLoader)
    model.eval()
    
    for i, batch in enumerate(valLoader) : 
        
        loss = WeakLoss(model, batch, args.softmaxMM)
        valLoss += loss.item()
        
    valLoss = valLoss / len(valLoader)
    msg = 'Epoch {:d}, Train Loss : {:.4f}, Val Loss : {:.4f}'.format(epoch, trainLoss , valLoss)
    with open(outHistory, 'w') as f :
        json.dump(history, f)
    print (msg)
    if valLoss < bestValLoss : 
        msg = 'Validation Loss Improved from {:.4f} to {:.4f}'.format(bestValLoss, valLoss)
        print (msg)
        bestValLoss = valLoss
        torch.save(model.state_dict(), outModel)

finalOut = os.path.join(args.outDir, 'netBest{:.3f}.pth'.format(bestValLoss))
cmd = 'mv {} {}',format(outModel, finalOut)
os.system(cmd)
        








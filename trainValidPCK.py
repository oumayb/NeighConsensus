import sys 
sys.path.append('./model')

import argparse 
import torch
import numpy as np
from model.model import NCNet
import torchvision.transforms as transforms
from dataloader import TrainLoader
from eval import evalPascal
from loss import WeakLoss
import torch.optim as optim
import pandas as pd
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
parser.add_argument('--testCSV', type=str, default = 'data/pf-pascal/test.csv', help='train csv')
parser.add_argument('--maxTestSize', type=int, default = 1000, help='max size in the test image')
parser.add_argument('--imgSize', type=int, default = 400, help='max size in the test image')


## learning parameter
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batchSize', type=int, default=16, help='batch size')
parser.add_argument('--nbEpoch', type=int, default=30, help='number of training epochs')
parser.add_argument('--neighConsKernel', nargs='+', type=int, default=[5,5,5], help='kernels sizes in neigh. cons.')
parser.add_argument('--neighConsChannel', nargs='+', type=int, default=[16,16,1], help='channels in neigh. cons')
parser.add_argument('--finetuneFeatExtractor', action='store_true', help='whether fine-tuning feature extractor')
parser.add_argument('--featExtractor', type=str, default='ResNet18Conv4', choices=['ResNet18Conv4', 'ResNet18Conv5'], help='feature extractor')
parser.add_argument('--cuda', action='store_true', help='GPU setting')
parser.add_argument('--softmaxMM', action='store_true', help='whether use softmax Mutual Matching')
parser.add_argument('--scoreTH', type=float, default=0.5, help='threshold of score to visualize matched grid')
parser.add_argument('--sigma', type=float, default=0.01, help='sigma in the spatial term to compute weight')
parser.add_argument('--nbWeightPoint', type=int, default=4, help='number of point to do interpolation')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha in the PCK metric')


args = parser.parse_args()
print(args)


## Set seed 
torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)
else : 
    raise RuntimeError('CPU Version is not supported yet.')
np.random.seed(1)

## Initial Model
model = NCNet(kernel_sizes=args.neighConsKernel, 
              channels=args.neighConsChannel, 
              featExtractor = args.featExtractor, 
              featExtractorPth = args.featExtractorPth, 
              finetuneFeatExtractor = args.finetuneFeatExtractor,
              softmaxMutualMatching = args.softmaxMM)

if not args.finetuneFeatExtractor:
    msg = '\nIgnore the gradient for the parameters in the feature extractor'
    print (msg)
    for p in model.featExtractor.parameters(): 
        p.requires_grad=False
        


if args.resumePth : 
    msg = '\nResume from {}'.format(args.resumePth)
    model.load_state_dict(torch.load(args.resumePth))
    
if args.cuda : 
    model.cuda()
    
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15], gamma=0.02)
## Train Test DataLoader
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
trainTransform = transforms.Compose([transforms.RandomResizedCrop(args.imgSize),
                                     transforms.ToTensor(), 
                                     normalize,])

testTransform = transforms.Compose([transforms.ToTensor(), 
                                     normalize,])


    
    
# pck results
df = pd.read_csv(args.testCSV)

trainLoader = TrainLoader(batchSize=args.batchSize, 
                          pairCSV=args.trainCSV, 
                          imgDir = args.imgDir, 
                          trainTransform = trainTransform)
                          

if not os.path.exists(args.outDir) : 
    os.mkdir(args.outDir)
    
    
# Train
bestTestPCK = 0.
history = {'TrainLoss' : [], 'TestPCK' : []}
outHistory = os.path.join(args.outDir, 'history.json')
outModel = os.path.join(args.outDir, 'netBest.pth')
    
for epoch in range(1, args.nbEpoch + 1) : 
    trainLoss = 0.
    lrScheduler.step()
    for i, batch in enumerate(trainLoader) : 
        
        optimizer.zero_grad()
        if args.cuda : 
            batch['source_image'] = batch['source_image'].cuda()
            batch['target_image'] = batch['target_image'].cuda()
        
        loss = WeakLoss(model, batch, args.softmaxMM)
        loss.backward()
        
        optimizer.step()
        
        trainLoss += loss.item()
        
        if i % 30 == 29 : 
            msg = '\nEpoch {:d}, Batch {:d}, Train Loss : {:.4f}'.format(epoch, i + 1, trainLoss / (i + 1))
            print (msg)
            
    ## Validation 
    trainLoss = trainLoss / len(trainLoader)
    
    model.eval()
    
    with torch.no_grad() : 
        pckRes = evalPascal(model, testTransform, df, args.featExtractor, args.softmaxMM, args.imgDir, args.maxTestSize, args.scoreTH, args.sigma, args.nbWeightPoint, args.alpha, None)
        
    testPCK = np.sum(pckRes * (pckRes >= 0).astype(np.float)) / np.sum(pckRes >= 0)
    msg = 'Epoch {:d}, Train Loss : {:.4f}, Test PCK : {:.4f}'.format(epoch, trainLoss , testPCK)
    with open(outHistory, 'w') as f :
        json.dump(history, f)
    print (msg)
    if testPCK > bestTestPCK : 
        msg = 'Test PCK Improved from {:.4f} to {:.4f}'.format(bestTestPCK, testPCK)
        print (msg)
        bestTestPCK = testPCK
        torch.save(model.state_dict(), outModel)
    model.train()

finalOut = os.path.join(args.outDir, 'netBest{:.3f}.pth'.format(bestTestPCK))
cmd = 'mv {} {}'.format(outModel, finalOut)
os.system(cmd)
        








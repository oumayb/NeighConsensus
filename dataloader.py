import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import pandas as pd
import numpy as np

class ImagePairDataset(Dataset):

    def __init__(self, pairCSV, imgDir, transform):
        df = pd.read_csv(pairCSV)
        self.sourceImg = df.iloc[:,0]
        self.targetImg = df.iloc[:,1]
        self.imgDir = imgDir         
        self.transform = transform
        self.nbPair = len(self.sourceImg)
              
    def __len__(self):
        return self.nbPair

    def __getitem__(self, idx):
        # get images
        sourceImg = Image.open(os.path.join(self.imgDir, self.sourceImg[idx])).convert('RGB')
        targetImg = Image.open(os.path.join(self.imgDir, self.targetImg[idx])).convert('RGB')
        if np.random.rand() > 0.5 : 
            sourceImg = sourceImg.transpose(method=Image.FLIP_LEFT_RIGHT)
            targetImg = targetImg.transpose(method=Image.FLIP_LEFT_RIGHT)
            
        sourceImg = self.transform(sourceImg)
        targetImg = self.transform(targetImg)
        
        sample = {'source_image': sourceImg, 'target_image': targetImg}
        
        return sample


def TrainLoader(batchSize, pairCSV, imgDir, trainTransform) : 
    
    dataloader = DataLoader(ImagePairDataset(pairCSV, imgDir, trainTransform), batch_size=batchSize, shuffle=True, droplast=True)
    
    return dataloader

def ValLoader(batchSize, pairCSV, imgDir, valTransform) : 
    
    dataloader = DataLoader(ImagePairDataset(pairCSV, imgDir, valTransform), batch_size=batchSize, shuffle=False)
    
    return dataloader
    
if __name__ == '__main__' : 
    
    import torchvision.transforms as transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    trainTransform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(), 
                                         normalize,])
    
    valTransform = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(), 
                                         normalize,])
    
    trainLoader = TrainLoader(batchSize=4, pairCSV='data/pf-pascal/train.csv', imgDir = 'data/pf-pascal/JPEGImages/', trainTransform = trainTransform)
    for data in trainLoader : 
        print (data['source_image'].size(), data['target_image'].size())
        break
        
    valLoader = ValLoader(batchSize=4, pairCSV='data/pf-pascal/val.csv', imgDir = 'data/pf-pascal/JPEGImages/', valTransform = valTransform)
    for data in valLoader : 
        print (data['source_image'].size(), data['target_image'].size())
        break
        

import os
import PIL.Image as Image
import numpy as np

## Function resize
## resize img, the largest dimension is maxSize
def resizeImg(I, minNet, strideNet, maxSize) :

    w, h = I.size
    
    wratio, hratio = w / maxSize, h / maxSize
    resizeRatio = max(wratio, hratio)
    
    w, h= w / resizeRatio, h / resizeRatio
    
    resizeW = round((w - minNet) / strideNet) * strideNet  + minNet
    resizeH = round((h - minNet) / strideNet) * strideNet + minNet
    
    
    return I.resize((resizeW-1, resizeH-1))
    
    
def getIndexMatchGT(df, imgDir, minNet, strideNet, maxSize, index) : 
    imgA = df['source_image'][ index ]
    imgB = df['target_image'][ index ]

    IA = Image.open(os.path.join(imgDir, imgA)).convert('RGB')
    wA, hA = IA.size
    IB = Image.open(os.path.join(imgDir, imgB)).convert('RGB')
    wB, hB = IB.size

    ## take coordinate and normalize between 0 and 1
    xA = np.array(list(map(float, df['XA'][ index ].split(';')))) 
    yA = np.array(list(map(float, df['YA'][ index ].split(';')))) 
    xB = np.array(list(map(float, df['XB'][ index ].split(';')))) 
    yB = np.array(list(map(float, df['YB'][ index ].split(';')))) 
    
    # compute PCK reference length refPCK (equal to max bounding box side in image_A)
    refPCK = max(xA.max() - xA.min(), yA.max() - yA.min())
    
    xA, yA, xB, yB = xA / wA, yA / hA, xB / wB, yB / hB

    IA = resizeImg (IA, minNet, strideNet, maxSize)
    IB = resizeImg (IB, minNet, strideNet, maxSize)
    return IA, IB, xA, yA, xB, yB, refPCK, wA, hA, wB, hB
    

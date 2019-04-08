import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def plotCorres(IA, IB, xA, yA, xB, yB, score = [], scoreTH = 0.5, lineColor = 'green', saveFig = 'toto.jpg') : 
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    wA, hA = IA.size
    wB, hB = IB.size
    nbPoint = len(xA)
    
    ax1.imshow(np.array(IA))
    ax2.imshow(np.array(IB))

    for i in range(nbPoint) : 
        if len(score) >0 and score[i] < scoreTH :
            continue
        xyA = (int(xA[i] * wA),int(yA[i] * hA))
        xyB = (int(xB[i] * wB),int(yA[i] * hB))

        con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=lineColor, linewidth = 3)
        ax2.add_artist(con)

        ax1.plot(xyA[0],xyA[1],'ro',markersize=5)
        ax2.plot(xyB[0],xyB[1],'ro',markersize=5)
    plt.savefig(saveFig, bbox_inches='tight')
    plt.close(fig)
    
    

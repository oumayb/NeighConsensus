python train.py --outDir pascalConv4Org --featExtractor ResNet18Conv4 --cuda
python train.py --outDir pascalConv5Org --featExtractor ResNet18Conv5 --cuda
python train.py --outDir pascalConv4Softmax --featExtractor ResNet18Conv4 --cuda --softmaxMM
python train.py --outDir pascalConv5Softmax --featExtractor ResNet18Conv5 --cuda --softmaxMM



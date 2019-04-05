wget https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
unzip PF-dataset-PASCAL.zip 'PF-dataset-PASCAL/JPEGImages/*'
rm -r PF-dataset-PASCAL.zip
mv PF-dataset-PASCAL/JPEGImages/ JPEGImages
rm -r PF-dataset-PASCAL

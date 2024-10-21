conda create -n lightning -y
conda activate lightning
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 lightning tensorboard opencv torchgeo pillow geopandas geopy -c pytorch -c nvidia -y
#https://github.com/tensorflow/tensorboard/issues/6874

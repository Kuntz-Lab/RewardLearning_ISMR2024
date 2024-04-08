Representation Learning for Point Clouds
====================
This repository has codes to run point cloud autoencoder 

# Installation and Documentation
Install EMD loss Pytorch: https://github.com/daerduoCarey/PyTorchEMD


# Run code
* Train autoencoder:
```sh
python training_AE.py
```
 Change the weight of the 2 loss components here `loss = loss_1*20 + loss_2 `. Make it so that the initial Huber loss = 1/5 * initial Chamfer loss.

` model = AutoEncoder(num_points=512*3)`: num_points is number of points of the reconstructed point cloud. Keep it small first (like 256) to test if the Autoencoder even works.


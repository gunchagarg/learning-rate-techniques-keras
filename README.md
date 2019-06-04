# learning-rate-techniques-keras
Exploring learning rates to improve model performance

Transfer Learning is a proven method to generate much better results in computer vision tasks. Most of the pretrained architectures (Resnet, VGG, inception, etc.) are trained on ImageNet and depending on the similarity of your data to the images on ImageNet, these weights will need to be altered more or less greatly.

## Reference


## Differential Learning
The phrase 'Differential Learning' implies the use of different learning rates on different parts of the network.

![alt text](https://cdn-images-1.medium.com/max/1200/1*4zrt6IeIhv55mUskGhXR7Q.png)

When it comes to modifying weights, the last layers of the model will often need the most changing, while deeper levels that are already well trained to detecting basic features (such as edges and outlines) will need less.

## Stochastic Gradient Descent with Restarts

It makes sense to reduce the learning rate as the training progresses, such that the algorithm does not overshoot and settles as close to the minimum as possible. With cosine annealing, we can decrease the learning rate following a cosine function.

SGDR is a recent variant of learning rate annealing that was introduced by Loshchilov & Hutter [5] in their paper "Sgdr: Stochastic gradient descent with restarts". In this technique, we increase the learning rate suddenly from time to time. Below is an example of resetting learning rate for three evenly spaced intervals with cosine annealing.

![alt text](https://cdn-images-1.medium.com/max/600/1*3kkV66xEObjWpYiGdBBivg.png)

The rationale behind suddenly increasing the learning rate is that, on doing so, the gradient descent does not get stuck at any local minima and may "hop" out of it in its way towards a global minimum.
Each time the learning rate drops to it's minimum point (every 100 iterations in the figure above), we call this a cycle. The authors also suggest making each next cycle longer than the previous one by some constant factor.

![alt text](https://cdn-images-1.medium.com/max/600/1*nBTMGa3WqhS2Iq4gCeCZww.png)

## Functionality Test

[dlr_implementation.py](dlr_implementation.py)
Modified source code of Adam optimizer to implement differential learning.

[sgdr_implementation.py](sgdr_implementation.py)
Implementation of SGDR using Keras callbacks.

[test.py](test.py)
Trains ResNet50 model on CIFAR 10 dataset with the use of differential learning and SGDR.

# The inception-resnet-v2 models trained from scratch via torch 

For personal interests, I and my friend Sunghun Kang (shuni@kaist.ac.kr) trained inception-resnet-v2 (http://arxiv.org/abs/1602.07261) from scratch based on torch, esp. facebook's training scripts (https://github.com/facebook/fb.resnet.torch)

I uploaded the torch model definition and the training script I used as PR in here (https://github.com/facebook/fb.resnet.torch). 

# Results

1. model figure

2. log comparisions

3. tools to parse log

# Notes

1. There seems typos in the paper (http://arxiv.org/abs/1602.07261)
  1. In Figure 17. the number of features in the last 1x1 convolution layer for residual path (= 1154) does not fit to the one of the output of the reduction A layer (= 1152); therefore, I changed the number of features in the last 1x1 conv to 1152. 

  2. In Figure 18. the number of features in 3x3 conv in the the second 1x1 conv -> 3x3 conv path (= 288) and the ones of 3x3 convs in the last path (1x1 conv -> 3x3 conv -> 3x3 conv) (= 288, 320) do not fit to the number of features in the following Inception-ResNet-C layer (= 2048); therefore, I changed them based on the model in https://gist.github.com/revilokeb/ab1809954f69d6d707be0c301947b69e

2. As mentioned in the inception-v4 paper (section 3.3.), scaling down by multiplying a scalar (0.1 or somewhat equivalent) to the last neuronal activities in residual path (the activities of linear conv of residual path) for residual layer seems very important to avoid explosion.
  * Since weights are usually normalized with unit-norm, the range of output values of linear (conv) layer becomes approximately (-1, 1) (if the output values of previous layers are normalized properly with batch-normalization)
  * At the initial stages in training, the addition of some-what random values with the range of (-1, 1) (from residual path) instabilizes the output of residual layer; therefore, training does not go well. 
  * The batch-normalizing the final output of residual layer befure relu (and after identitiy + residual) seems not stabilize the output activities since the large contributions of the activies of residual path (see, https://github.com/revilokeb/inception_resnetv2_caffe)

3. I used the custom learning rate scheduling since 1) the batch size information wasn't provided in the inception v4 paper and 2) the custom lr scheduling has been worked well for differnt types of imagenet classifier models. 
  * What-so-called `step` style learning rate scheduling is used with sgd
  * This lr scheduling and its variant with different stepsize have been work properly with googlenet (inception-v1) and googlenet-bn (inception-v2)   * As far as the regularization of the inception-resnet-v2 are properly applied, the learning rate scheduling is expected to be applicable with inception-resnet-v2. 

4. The momentum for SGD was set to 0.4737, which is the equivalent value on torch style sgd for the 0.9 momentum on caffe style sgd (see, https://github.com/KaimingHe/deep-residual-networks).


# References 
1. http://arxiv.org/abs/1602.07261
2. https://github.com/facebook/fb.resnet.torch
3. https://github.com/revilokeb/inception_resnetv2_caffe
4. https://www.reddit.com/r/MachineLearning/comments/47asuj/160207261_inceptionv4_inceptionresnet_and_the/?
5. https://github.com/beniz/deepdetect/issues/89


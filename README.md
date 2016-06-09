# The inception-resnet-v2 models trained from scratch via torch 

For personal interests, I and my friend Sunghun Kang (shuni@kaist.ac.kr) trained inception-resnet-v2 (http://arxiv.org/abs/1602.07261) from scratch based on torch, esp. facebook's training scripts (https://github.com/facebook/fb.resnet.torch)

I uploaded the torch model definition and the training script I used as PR in [here](https://github.com/facebook/fb.resnet.torch/pull/64).

Because of limited computational resources we have, we tried only few training conditions. For someone who are interested in achieving the same performance in the paper, I added some notes we've learned throughout trials. Those might be helpful. 


## Requirements

0. See, https://github.com/facebook/fb.resnet.torch/pull/64 


## Settings

0. SGD, `batchsize = 32 x 2 = 64`, and learning rate scheduling with step-style where `stepsize = 12800` and `gamma = 0.96`
  * trained with Titan X x 2, each of which handles effective batchsize of 32 (this information matters since batch normalization in this script does not share their normalization constants). 
  * ended when `nEpochs = 90` (It took approximately 21 days. At least 300 epoches are required to match with the results in the original paper. See, note #5)

0. SGD, `batchsize = 32 x 2 = 64`, and learning rate scheduling with step-style where `stepsize = 25600` and `gamma = 0.96`
  * trained with Titan X x 2, each of which handles effective batchsize of 32. 
  * in progress.  


## Results

0. 1-crop validation error on ImageNet (center 299x299 crop from resized image with 328x328): 

  1. Single-crop (299x299) validation error rate

    | Network               | Top-1 error | Top-5 error |
    | --------------------- | ----------- | ----------- |
    | Setting 1             | 24.407      | 7.407       |
    | Setting 2             | N/A         | N/A         |

  2. Training curves on ImageNet (solid lines: 1-crop top-1 error; dashed lines: 1-crop top-5 error):
    * You can plot yourself based on the scripts in the `tools/plot_log.py`
    
    ![Training curves](https://github.com/lim0606/torch-inception-resnet-v2/blob/master/figures/b64_s12800_i1801710.png)

## Notes

0. There seems typos in the paper (http://arxiv.org/abs/1602.07261)
  0. In Figure 17. the number of features in the last 1x1 convolution layer for residual path (= 1154) does not fit to the one of the output of the reduction A layer (= 1152); therefore, I changed the number of features in the last 1x1 conv to 1152. 

  0. In Figure 18. the number of features in 3x3 conv in the the second 1x1 conv -> 3x3 conv path (= 288) and the ones of 3x3 convs in the last path (1x1 conv -> 3x3 conv -> 3x3 conv) (= 288, 320) do not fit to the number of features in the following Inception-ResNet-C layer (= 2048); therefore, I changed them based on the model in https://gist.github.com/revilokeb/ab1809954f69d6d707be0c301947b69e

0. As mentioned in the inception-v4 paper (section 3.3.), scaling down by multiplying a scalar (0.1 or somewhat equivalent) to the last neuronal activities in residual path (the activities of linear conv of residual path) for residual layer seems very important to avoid explosion.
  * Since weights are usually normalized with unit-norm, the range of output values of linear (conv) layer becomes approximately (-1, 1) (if the output values of previous layers are normalized properly with batch-normalization)
  * At the initial stages in training, the addition of some-what random values with the range of (-1, 1) (from residual path) instabilizes the output of residual layer; therefore, training does not go well. 
  * The batch-normalizing the final output of residual layer before relu (and after identitiy + residual) seems not stabilize the output activities since the large contributions of the activies of residual path (see, https://github.com/revilokeb/inception_resnetv2_caffe)

0. I used the custom learning rate scheduling since 1) the batch size information wasn't provided in the inception v4 paper and 2) the custom lr scheduling has been worked well for differnt types of imagenet classifier models. 
  * So-called `step`-style learning rate scheduling is used with SGD 
  * This lr scheduling and its variant with different stepsize have been work properly with googlenet (inception-v1) and googlenet-bn (inception-v2)  
  * As far as the regularization of the inception-resnet-v2 are properly applied, the learning rate scheduling is expected to be applicable with inception-resnet-v2. 

0. The momentum for SGD was set to 0.4737, which is the equivalent value on torch style sgd for the 0.9 momentum on caffe style sgd (see, https://github.com/KaimingHe/deep-residual-networks).
  * Based on this info, the base learning rate should set as `0.045 * 1.9 = 0.0885`, but I just tried with `0.045`

0. Based on the comparison between the loss curve I got and the one in the paper, the effective batchsize in the original paper seems like 32.
  * Based on this guess, the equivalent setting for training should be with 1) `stepsize = 80076`, 2) `gamma = 0.94`, and 3) `nEpochs = 300` when `batchsize = 64`. (It will takes approximately 60 days with 2 Titan Xs)
  * Therefore, the decaying rate for lr of the learning rate scheduling I tried may be too fast.

## Models

0. For setting 1, see [Google Drive](https://drive.google.com/folderview?id=0By3GiE-Oc72rQlJSMVZ0Ri1pb1k&usp=sharing)

0. For setting 2. trainingn is in progress

## References 
0. http://arxiv.org/abs/1602.07261
0. https://github.com/facebook/fb.resnet.torch
0. https://github.com/revilokeb/inception_resnetv2_caffe
0. https://www.reddit.com/r/MachineLearning/comments/47asuj/160207261_inceptionv4_inceptionresnet_and_the/?
0. https://github.com/beniz/deepdetect/issues/89


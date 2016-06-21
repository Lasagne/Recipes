# Identity Mappings in Deep Residual Networks in Lasagne/Theano

Reproduction of some of the results from the recent [MSRA ResNet](https://arxiv.org/abs/1603.05027) paper and the follow-up [Wide-Resnet](https://arxiv.org/pdf/1605.07146v1.pdf) paper. Exploring the full-preactivation style residual layers, both normal and wide.

![PreResNet](https://qiita-image-store.s3.amazonaws.com/0/100523/a156a5c2-026b-de55-a6fb-e4fa1772b42c.png) ![WideResNet](http://i.imgur.com/3b0fw7b.png)

## Results on CIFAR-10

Results are presented as classification error percent.

| ResNet Type | Original Paper | Test Results |
| -----------|-----------|----------- |
| ResNet-110 | 6.37 | 6.38 |
| ResNet-164 | 5.46 | 5.66 |
| Wide-ResNet | 5.55 | 5.41 |

**Note:** ResNet-110 is the stacked 3x3 filter variant and ResNet-164 is the 'botttleneck' architecture. Both use the new pre-activation units as proposed in the paper. For Wide-ResNet the paper and test results are for depth 16 and width multiplier of 4. This repo also uses the preprocessing and training parameters from the Preactivation-ResNet paper and not the Wide-ResNet paper, so it is not a 1 to 1 comparison with the Wide-ResNet paper results

### ResNet-110

![ResNet-110](http://i.imgur.com/Y7VrxOC.png)

### ResNet-164

![ResNet-164](http://i.imgur.com/VznjI5x.png)

### Wide-ResNet Depth-16 Width-4

![Wide-ResNet](http://i.imgur.com/IuBppdJ.png)

## Implementation details

Had to use batch sizes of 64 for ResNet-110 and 48 for ResNet-164 due to hardware constraints. The data augmentation is exactly the same, only translations by padding then copping and left-right flipping. More details can be found [here](http://florianmuellerklein.github.io/wRN_vs_pRN/)

## Running the networks

To run your own PreResNet simply call train.py with system args defining the type and depth of the network.

```
train_nn.py [type] [depth] [width]
```

Testing the accuracy on the test set can be done in a very similar way.

```
test_model.py [type] [depth] [width]
```

-**Type (string)**:  Can be 'normal', 'bottleneck' or 'wide'

-**Depth (integer)**:  Serves as the multiplier for how many residual blocks to insert into each section of the network

-**Width (integer)**: Only for wide-ResNet, serves as the filter multiplier [3x3, 16*k] for residual blocks, excluding the first convolution layer.

| Group | Size | Multiplier |
| ------|:------:|:----------:|
| Conv1 | [3x3, 16] | - |
| Conv2 | [3x3, 16]<br>[3x3, 16] | N |
| Conv3 | [3x3, 32]<br>[3x3, 32] | N |
| Conv4 | [3x3, 64]<br>[3x3, 64] | N |
| Avg-Pool | 8x8 | - |
| Softmax  | 10 | - |

The extracted 'cifar-10-batches-py' from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz must be extracted into a 'data' folder within the working directory.


```
 PreResNet Directory
 |__ test_model.py
 |__ train_nn.py
 |__ models.py
 |__ utils.py
 |__ data
     |__cifar-10-batches-py
        |__ data_batch_1
        |__ data_batch_2
        |__ ...
```


**Note:** If using the wide-ResNet, the implementation in the [paper](https://arxiv.org/pdf/1605.07146v1.pdf) will be slightly different than the one here. They use different preprocessing and a different value for L2. This repo stays consistent with the [MSRA paper](https://arxiv.org/abs/1603.05027).

### References

* Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Identity Mappings in Deep Residual Networks", [link](https://arxiv.org/pdf/1603.05027v2.pdf)
* Sergey Zagoruyko, Nikos Komodakis, "Wide Residual Networks", [link](https://arxiv.org/pdf/1605.07146v1.pdf)

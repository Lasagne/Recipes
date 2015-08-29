# Image Captioning with LSTM

This is a partial implementation of "Show and Tell: A Neural Image Caption Generator" (http://arxiv.org/abs/1411.4555), borrowing heavily from Andrej Karpathy's NeuralTalk (https://github.com/karpathy/neuraltalk)

This example consists of three parts:
1. COCO Preprocessing - prepare the dataset by precomputing image representations using GoogLeNet
2. RNN Training - train a network to predict image captions
3. Caption Generation - use the trained network to caption new images
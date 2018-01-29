## AC-GAN
A tensorflow implementation of AC-GAN. 

Reference [sugartensor-ac-gan](https://github.com/buriburisuri/ac-gan)
## AC-GAN Structure
![](../../images/ac-gan-fig-01.png)

ac-gan build a gan which discriminator output include not only the probability of the real/fake but also the class label distribution.
[https://arxiv.org/pdf/1610.09585v3.pdf](https://arxiv.org/pdf/1610.09585v3.pdf)

# train
python train # 50 epoch

# test
python generate.py  #保存图片

WARNING:tensorflow:VARIABLES collection name is deprecated, please use GLOBAL_VARIABLES instead; VARIABLES will be removed after 2017-03-02.
[9 9 4 0 4 3 6 0 1 6 3 6 9 5 2 1 7 2 1 4 2 2 5 4 4 3 6 3 5 5 3 3 3 5 1 9 4
 0 5 1 4 3 8 4 8 0 8 3 5 3 7 9 3 2 8 0 0 0 4 9 6 0 6 7 2 1 9 1 4 6 7 2 1 5
 4 2 3 5 4 3 6 8 6 2 9 1 6 2 9 2 8 4 3 9 9 4 8 1 6 0]
2018-01-29 02:00:27.849251: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Sample image save to "./result/fake.png"
Sample image save to "./result/sample.png"

python disc_rate.py  # 辨别率

Extracting ./asset/data/mnist/train-images-idx3-ubyte.gz
Extracting ./asset/data/mnist/train-labels-idx1-ubyte.gz
Extracting ./asset/data/mnist/t10k-images-idx3-ubyte.gz
Extracting ./asset/data/mnist/t10k-labels-idx1-ubyte.gz
WARNING:tensorflow:VARIABLES collection name is deprecated, please use GLOBAL_VARIABLES instead; VARIABLES will be removed after 2017-03-02.
2018-01-29 01:44:55.906617: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
accruacy in 10000 test samples is  0.987600009441

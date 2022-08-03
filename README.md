# DExp_pruning

DExp way : Complete pruning using the expectation scaling factor

## Running Code

    In this code, you can run our models on CIFAR-10, CIFAR-100 and ImageNet dataset. The code has been tested by Python 3.6, Pytorch 1.6 and CUDA 10.2 on Windows 10.
    For the channel mask generation, no additional settings are required. You can just set the required parameters in main.py and it will run.

## parser
```shell
&&& main.py &&&
/data_dir/ : Dataset storage address
/dataset/ ： dataset - CIFAR10, CIFAR100 and Imagenet
/lr/ ： initial learning rate
/lr_decay_step/ ： learning rate decay step
/resume/ ： load the model from the specified checkpoint
/resume_mask/ ： After the program is interrupted, the task file can be used to continue running
/job_dir/ ： The directory where the summaries will be stored
/epochs/ ： The num of epochs to fine-tune
/start_cov/ ： The num of conv to start prune
/compress_rate/ ： compress rate of each conv
/arch/ ： The architecture to prune

&&& cal_flops_params.py &&&
/input_image_size/ : 32(CIFAR-10), 32(CIFAR-100),  224(ImageNet)
/arch/ ： The architecture to prune
/compress_rate/ ： compress rate of each conv
```

## Model Training

For the ease of reproducibility. we provide some of the experimental results and the corresponding pruned rate of every layer as belows:

### 1. VGG-16

| Flops     | Accuracy  |CIFAR-10/Model               |
|-----------|-----------|-----------------------------|
| 100%      | 93.96%    |[VGG16](https://drive.google.com/file/d/1q_uzAvsAPyQxdaeYWy9NkpnRxwWRr_zc/view?usp=sharing)
| -62.39%    | 93.47%    |[VGG](https://drive.google.com/file/d/1ZOc3ImLkqOew22fb9_HU_y_IIRmC8uAz/view?usp=sharing)|

### 2. ResNet-56

| Flops     | Accuracy  |CIFAR-10/Model               |
|-----------|-----------|-----------------------------|
| 100%      | 93.26%    |[ResNet-56](https://drive.google.com/file/d/1WE83j7rlKlCp-tslSL6hS-d_mJe4ZQ2r/view?usp=sharing)|
| -66.52%   | 93.19%    |[Res](https://drive.google.com/file/d/1WQCfy2B7FJOtY5b2BxBa5090OVKBtZ_T/view?usp=sharing)|
|-----------|-----------|-----------------------------|
| Flops     | Accuracy  |CIFAR-100/Model              |
| 100%      | 70.45%    |[ResNet-56](https://drive.google.com/file/d/1FBiXRMWZCuloK62E8XK-PMzAJwxZvO4Z/view?usp=sharing)|
| -54.67%   | 70.66%    |[Res](https://drive.google.com/file/d/1LmUX2z7d3cHDtETzRWpHCa7qbE1DibEj/view?usp=sharing)|

### 3. ResNet-110

| Flops     | Accuracy  |CIFAR-10/Model               | 
|-----------|-----------|-----------------------------|
| 100%      | 93.50%    |[ResNet-110](https://drive.google.com/file/d/1YhJHzSBiCsQcNIdamI2_GzclpXvSXcPG/view?usp=sharing)|
|-----------|-----------|-----------------------------|
| Flops     | Accuracy  |CIFAR-100/Model              | 
| 100%      | 70.37%    |[ResNet-110](https://drive.google.com/file/d/1WPnotBfIrV1T0xwsd3yigZXOiZgyJNin/view?usp=sharing)|
| -66.89%    | 71.32%   |[Res](https://drive.google.com/file/d/1NaGZfNApo8J0z9BOXl2pFdv2QXZ5GNV1/view?usp=sharing)|

### 4. GoogLeNet

| Flops     | Accuracy  |CIFAR-10/Model               |
|-----------|-----------|-----------------------------|
| 100%      | 95.05%    |[GoogLeNet](https://drive.google.com/file/d/1TXF2OUwkUUWBVAj5Q-QRRO2ZNVRcdmqB/view?usp=sharing)| 
| -71.05%    | 94.75%    |[GoogLeNet](https://drive.google.com/file/d/18EDQ7nVcUpTVSIbObrNgaooPAL-Z9Wvl/view?usp=sharing)| 
|-----------|-----------|-----------------------------|
| Flops     | Accuracy  |CIFAR-100/Model              |
| 100%      | 76.57%    |[GoogLeNet](https://drive.google.com/file/d/1aUWIhr3NfMrQyKeb8iAwrtr5XLQ48CtW/view?usp=sharing)| 
| -71.05%    | 76.77%    |[GoogLeNet](https://drive.google.com/file/d/1HPnoZNWdsvhvzE0D8ZEQNAPyhBlUufVN/view?usp=sharing)| 

### 5. ResNet-50

| Flops     | Accuracy  |ImageNet/Model                |
|-----------|-----------|-----------------------------|
| 100%      | 76.15%    |[ResNet-50](https://drive.google.com/file/d/1H8MlYJCSLmjJOaLjSBMCeh5zfN2bEYT9/view?usp=sharing)
| -58.43%   | 75.00%    |[ResNet-50](https://drive.google.com/file/d/1PPToJ7QR6iKcJF9agye530by6rbMwu0V/view?usp=sharing)|


### 5. ResNet-18

| Flops     | Accuracy  |ImageNet/Model                |
|-----------|-----------|-----------------------------|
| 100%      | 69.66%    |[ResNet-18](https://drive.google.com/file/d/1S3Tm7KnvgKCrby2aG0r3rbHbq-mMJsNw/view?usp=sharing)
| -57.69%   | 64.48%    |[ResNet-18](https://drive.google.com/file/d/1M8ZD1t-4QIyV6Gtnf1DlyFIAyOCij70u/view?usp=sharing)|




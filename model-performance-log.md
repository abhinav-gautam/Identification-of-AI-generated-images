# Model Performance Log

This log it to track trained model's performance and architectures. This will be helpful for tracking model's performance and improvements.

## Model Performance

| S. No. | Model Name                                  | Training Dataset | Architecture      | Activation Function | Epoch | Optimizer | Learning Rate | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| ------ | ------------------------------------------- | ---------------- | ----------------- | ------------------- | ----- | --------- | ------------- | ------------- | ----------------- | --------------- | ------------------- |
| 1.     | advance_model_ai-art                        | AI Art           | Advance Model     | relu                | 10    | Adam      | 0.001         | 0.3478        | 0.8611            | 0.4645          | 0.8080              |
| 2.     | advance_model_cifake                        | CIFAKE           | Advance Model     | relu                | 10    | Adam      | 0.001         | 0.3291        | 0.8623            | 0.4437          | 0.8462              |
| 3.     | alexnet_model_cifake                        | CIFAKE           | AlexNet           | relu                | 10    | Adam      | 0.001         | 0.6932        | 0.5003            | 0.6931          | 0.5000              |
| 4.     | basic_model_ai-art                          | AI Art           | Basic Model       | relu                | 10    | Adam      | 0.001         | 0.3819        | 0.8480            | 0.3862          | 0.8361              |
| 5.     | basic_model_cifake                          | CIFAKE           | Basic Model       | relu                | 10    | Adam      | 0.001         | 0.2816        | 0.8814            | 0.3470          | 0.8648              |
| 6.     | basic_model_merged                          | CIFAKE + AI Art  | Basic Model       | relu                | 10    | Adam      | 0.001         | 0.3161        | 0.8689            | 0.4446          | 0.8171              |
| 7.     | basic-tanh_model_cifake                     | CIFAKE           | Basic Model       | tanh                | 100   | Adam      | 0.001         | 0.4088        | 0.8154            | 0.4040          | 0.8231              |
| 8.     | inception-v3_model_ai-art                   | Ai Art           | InceptionV3       | relu                | 10    | Adam      | 0.001         | 0.3762        | 0.8483            | 0.4263          | 0.8290              |
| 9.     | inception-v3_model_cifake                   | CIFAKE           | InceptionV3       | relu                | 10    | Adam      | 0.001         | 0.4065        | 0.8147            | 0.3596          | 0.8418              |
| 10.    | inception-v3-dense_model_cifake             | CIFAKE           | InceptionV3 Dense | relu                | 10    | Adam      | 0.001         | 0.3952        | 0.8211            | 0.3605          | 0.8422              |
| 11.    | inception-v3-dense-epochs100_model_cifake   | CIFAKE           | InceptionV3 Dense | relu                | 100   | Adam      | 0.001         | 0.3519        | 0.8474            | 0.3185          | 0.8649              |
| 12.    | lenet5_model_ai-art                         | Ai Art           | LeNet5            | relu                | 10    | Adam      | 0.001         | 0.4404        | 0.8253            | 0.4247          | 0.8253              |
| 13.    | lenet5_model_cifake                         | CIFAKE           | LeNet5            | relu                | 10    | Adam      | 0.001         | 0.4181        | 0.8081            | 0.3678          | 0.8358              |
| 14.    | lenet5-dense_model_ai-art                   | Ai Art           | LeNet5 Dense      | relu                | 10    | Adam      | 0.001         | 0.4886        | 0.8247            | 0.4750          | 0.8253              |
| 15.    | lenet5-dense_model_cifake                   | CIFAKE           | LeNet5 Dense      | relu                | 10    | Adam      | 0.001         | 0.4565        | 0.7878            | 0.3578          | 0.8388              |
| 16.    | lenet5-epochs100_model_cifake               | CIFAKE           | LeNet5 Dense      | relu                | 100   | Adam      | 0.001         | 0.3698        | 0.8391            | 0.3793          | 0.8414              |
| 17.    | resnet50_model_ai-art                       | Ai Art           | ResNet50          | relu                | 10    | Adam      | 0.001         | 0.4344        | 0.8210            | 1.0570          | 0.8274              |
| 18.    | resnet50_model_cifake                       | CIFAKE           | ResNet50          | relu                | 10    | Adam      | 0.001         | 0.5493        | 0.7237            | 0.9011          | 0.6001              |
| 19.    | resnet50-regularized_model_cifake           | CIFAKE           | ResNet50          | relu                | 10    | Adam      | 0.001         | 0.5140        | 0.7812            | 0.4894          | 0.8000              |
| 20.    | resnet50-regularized-epochs100_model_cifake | CIFAKE           | ResNet50          | relu                | 100   | Adam      | 0.001         | 0.4834        | 0.7910            | 0.4734          | 0.7982              |
| 21.    | vgg16_model_ai-art                          | Ai Art           | VGG16             | relu                | 10    | Adam      | 0.001         | 0.3549        | 0.8577            | 0.4043          | 0.8369              |
| 22.    | vgg16_model_cifake                          | CIFAKE           | VGG16             | relu                | 10    | Adam      | 0.001         | 0.3480        | 0.8471            | 0.3010          | 0.8722              |
| 23.    | vgg16_model_merged                          | CIFAKE + AI Art  | VGG16             | relu                | 10    | Adam      | 0.001         | 0.3932        | 0.8227            | 0.4068          | 0.8288              |
| 24.    | vgg16-dense_model_cifake                    | CIFAKE           | VGG16 Dense       | relu                | 10    | Adam      | 0.001         | 0.3482        | 0.8464            | 0.3139          | 0.8646              |
| 25.    | vgg16-epochs100_model_cifake                | CIFAKE           | VGG16             | relu                | 100   | Adam      | 0.001         | 0.2984        | 0.8719            | 0.2852          | 0.8835              |
| 26.    | vgg16-epochs100-tanh_model_cifake           | CIFAKE           | VGG16             | tanh                | 100   | Adam      | 0.001         | 0.3298        | 0.8560            | 0.3043          | 0.8693              |
| 27.    | vgg16-lr0.01_model_cifake                   | CIFAKE           | VGG16             | relu                | 10    | Adam      | 0.01          | 0.4019        | 0.8209            | 0.3708          | 0.8403              |
| 28.    | vgg16-lr0.01-epochs100_model_cifake         | CIFAKE           | VGG16             | relu                | 100   | Adam      | 0.01          | 0.4043        | 0.8190            | 0.3758          | 0.8352              |
| 29.    | vgg16-regularized-adamax_model_cifake       | CIFAKE           | VGG16             | relu                | 19    | Adamax    | 0.001         | 0.3290        | 0.8900            | 0.3500          | 0.8786              |
| 30.    | vgg16-regularized-epochs100_model_cifake    | CIFAKE           | VGG16             | relu                | 100   | Adam      | 0.001         | 0.3719        | 0.8616            | 0.3573          | 0.8687              |
| 31.    | vgg16-rmsprop-epochs100_model_cifake        | CIFAKE           | VGG16             | relu                | 100   | RMSProp   | 0.001         | 0.4407        | 0.8214            | 0.4248          | 0.8407              |
| 32.    | vgg16-sgd_model_cifake                      | CIFAKE           | VGG16             | relu                | 10    | SGD       | 0.01          | 0.3832        | 0.8270            | 0.3382          | 0.8526              |
| 33.    | vgg16-sgd-epochs100_model_cifake            | CIFAKE           | VGG16             | relu                | 100   | SGD       | 0.01          | 0.3189        | 0.8609            | 0.2850          | 0.8783              |

| S. No. | Model Name                                         | Training Dataset | Architecture | Optimizer | Training Loss | Training Accuracy | Training Precision | Training Recall | Validation Loss | Validation Accuracy | Validation Precision | Validation Recall |
| ------ | -------------------------------------------------- | ---------------- | ------------ | --------- | ------------- | ----------------- | ------------------ | --------------- | --------------- | ------------------- | -------------------- | ----------------- |
| 1.     | resnet50-regularized-adamax_model_cifake           | CIFAKE           | ResNet50     | Adamax    | 0.5592        | 0.7377            | 0.7397             | 0.7334          | 0.5142          | 0.7700              | 0.8188               | 0.6935            |
| 2.     | resnet50-regularized-adamax-noaug_model_cifake     | CIFAKE           | ResNet50     | Adamax    | 0.0382        | 0.9940            | 0.9940             | 0.9943          | 0.1805          | 0.9574              | 0.9519               | 0.9635            |
| 4.     | resnet50-regularized-adamax-noaug_seq-model_cifake | CIFAKE           | ResNet50     | Adamax    | 0.0346        | 0.9938            | 0.9942             | 0.9933          | 0.1070          | 0.9667              | 0.9654               | 0.9681            |
| 5.     | resnet50-regularized-adamax-rescale_model_cifake   | CIFAKE           | ResNet50     | Adamax    | 0.6950        | 0.5005            | 0.5004             | 0.0             | 0.5             | 0.0                 | 0.0                  | 0.0               |
| 6.     | resnet50-regularized-adamax_model_ai-art           | AI Art           | ResNet50     | Adamax    | 0.4658        | 0.8317            | 0.5894             | 0.1220          | 91.5031         | 0.8253              | 0.0                  | 0.0               |
| 7.     | resnet50-regularized-adamax-noaug_model_ai-art     | AI Art           | ResNet50     | Adamax    | 0.1256        | 0.9709            | 0.9342             | 0.8968          | 3.6233          | 0.1746              | 0.1746               | 1.0               |
| 8.     | vgg16-regularized-adamax_model_ai-art              | AI Art           | VGG16        | Adamax    | 0.3718        | 0.8504            | 0.6441             | 0.3214          | 0.4318          | 0.8364              | 0.5895               | 0.2090            |
| 9.     | vgg16-regularized-adamax_model_merged              | CIFAKE + AI Art  | VGG16        | Adamax    | 0.2016        | 0.9273            | 0.9378             | 0.8989          | 0.4153          | 0.8555              | 0.9925               | 0.6783            |
| 10.    | vgg16-regularized-adamax-noaug_model_ai-art        | AI Art           | VGG16        | Adamax    | 0.3742        | 0.8586            | 0.6670             | 0.3809          | 1.3096          | 0.1746              | 0.1746               | 1.0               |
| 11.    | vgg16-regularized-adamax-noaug_model_cifake        | CIFAKE           | VGG16        | Adamax    | 0.0251        | 0.9956            | 0.9960             | 0.9950          | 0.1790          | 0.9603              | 0.9379               | 0.9859            |
| 12.    | vgg16-regularized-adam-noaug_model_cifake          | CIFAKE           | VGG16        | Adam      | 0.0438        | 0.9886            | 0.9897             | 0.9875          | 0.1909          | 0.9475              | 0.9205               | 0.9797            |

## Model Architectures

### Basic Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 30, 30, 100)       2800

 max_pooling2d (MaxPooling2D  (None, 15, 15, 100)      0
 )

 conv2d_1 (Conv2D)           (None, 13, 13, 100)       90100

 max_pooling2d_1 (MaxPooling  (None, 6, 6, 100)        0
 2D)

 flatten (Flatten)           (None, 3600)              0

 dropout (Dropout)           (None, 3600)              0

 dense (Dense)               (None, 50)                180050

 dense_1 (Dense)             (None, 2)                 102

=================================================================
Total params: 273,052
Trainable params: 273,052
Non-trainable params: 0
_________________________________________________________________
```

### Advance Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_40 (Conv2D)          (None, 30, 30, 32)        896

 max_pooling2d_40 (MaxPoolin  (None, 15, 15, 32)       0
 g2D)

 batch_normalization_64 (Bat  (None, 15, 15, 32)       128
 chNormalization)

 conv2d_41 (Conv2D)          (None, 13, 13, 64)        18496

 max_pooling2d_41 (MaxPoolin  (None, 6, 6, 64)         0
 g2D)

 batch_normalization_65 (Bat  (None, 6, 6, 64)         256
 chNormalization)

 conv2d_42 (Conv2D)          (None, 4, 4, 128)         73856

 max_pooling2d_42 (MaxPoolin  (None, 2, 2, 128)        0
 g2D)

 batch_normalization_66 (Bat  (None, 2, 2, 128)        512
 chNormalization)

 global_average_pooling2d_10  (None, 128)              0
  (GlobalAveragePooling2D)

 dense_32 (Dense)            (None, 1024)              132096

 batch_normalization_67 (Bat  (None, 1024)             4096
 chNormalization)

 dense_33 (Dense)            (None, 512)               524800

 batch_normalization_68 (Bat  (None, 512)              2048
 chNormalization)

 dense_34 (Dense)            (None, 256)               131328

 batch_normalization_69 (Bat  (None, 256)              1024
 chNormalization)

 dropout_8 (Dropout)         (None, 256)               0

 dense_35 (Dense)            (None, 2)                 514

=================================================================
Total params: 890,050
Trainable params: 886,018
Non-trainable params: 4,032
_________________________________________________________________
```

### AlexNet Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_7 (Conv2D)           (None, 55, 55, 96)        34944

 max_pooling2d_4 (MaxPooling  (None, 27, 27, 96)       0
 2D)

 conv2d_8 (Conv2D)           (None, 27, 27, 256)       614656

 max_pooling2d_5 (MaxPooling  (None, 13, 13, 256)      0
 2D)

 conv2d_9 (Conv2D)           (None, 13, 13, 384)       885120

 conv2d_10 (Conv2D)          (None, 13, 13, 384)       1327488

 conv2d_11 (Conv2D)          (None, 13, 13, 256)       884992

 max_pooling2d_6 (MaxPooling  (None, 6, 6, 256)        0
 2D)

 flatten_1 (Flatten)         (None, 9216)              0

 dropout_2 (Dropout)         (None, 9216)              0

 dense_3 (Dense)             (None, 4096)              37752832

 dropout_3 (Dropout)         (None, 4096)              0

 dense_4 (Dense)             (None, 4096)              16781312

 dense_5 (Dense)             (None, 2)                 8194

=================================================================
Total params: 58,289,538
Trainable params: 58,289,538
Non-trainable params: 0
_________________________________________________________________
```

### LeNet5 Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_22 (Conv2D)          (None, 28, 28, 12)        912

 average_pooling2d_20 (Avera  (None, 14, 14, 12)       0
 gePooling2D)

 conv2d_23 (Conv2D)          (None, 10, 10, 32)        9632

 average_pooling2d_21 (Avera  (None, 5, 5, 32)         0
 gePooling2D)

 conv2d_24 (Conv2D)          (None, 1, 1, 240)         192240

 flatten_3 (Flatten)         (None, 240)               0

 dense_16 (Dense)            (None, 168)               40488

 dense_17 (Dense)            (None, 2)                 338

=================================================================
Total params: 243,610
Trainable params: 243,610
Non-trainable params: 0
_________________________________________________________________
```

### LeNet5 Dense Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_31 (Conv2D)          (None, 28, 28, 12)        912

 average_pooling2d_26 (Avera  (None, 14, 14, 12)       0
 gePooling2D)

 conv2d_32 (Conv2D)          (None, 10, 10, 32)        9632

 average_pooling2d_27 (Avera  (None, 5, 5, 32)         0
 gePooling2D)

 conv2d_33 (Conv2D)          (None, 1, 1, 240)         192240

 flatten_6 (Flatten)         (None, 240)               0

 dense_24 (Dense)            (None, 1024)              246784

 dense_25 (Dense)            (None, 2)                 2050

=================================================================
Total params: 451,618
Trainable params: 451,618
Non-trainable params: 0
_________________________________________________________________
```

### ResNet50 Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 resnet50 (Functional)       (None, 1, 1, 2048)        23587712

 global_average_pooling2d_1   (None, 2048)             0
 (GlobalAveragePooling2D)

 dense_4 (Dense)             (None, 1024)              2098176

 batch_normalization_3 (Batc  (None, 1024)             4096
 hNormalization)

 dense_5 (Dense)             (None, 512)               524800

 batch_normalization_4 (Batc  (None, 512)              2048
 hNormalization)

 dense_6 (Dense)             (None, 256)               131328

 batch_normalization_5 (Batc  (None, 256)              1024
 hNormalization)

 dropout_1 (Dropout)         (None, 256)               0
...
Total params: 26,349,698
Trainable params: 26,292,994
Non-trainable params: 56,704
_________________________________________________________________
```

### Inception V3 Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 resizing_1 (Resizing)       (None, 75, 75, None)      0

 inception_v3 (Functional)   (None, 1, 1, 2048)        21802784

 global_average_pooling2d_1   (None, 2048)             0
 (GlobalAveragePooling2D)

 dense_2 (Dense)             (None, 512)               1049088

 dense_3 (Dense)             (None, 2)                 1026

=================================================================
Total params: 22,852,898
Trainable params: 1,050,114
Non-trainable params: 21,802,784
_________________________________________________________________
```

### Inception V3 Dense Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 resizing_3 (Resizing)       (None, 75, 75, None)      0

 inception_v3 (Functional)   (None, 1, 1, 2048)        21802784

 global_average_pooling2d_3   (None, 2048)             0
 (GlobalAveragePooling2D)

 dense_8 (Dense)             (None, 1024)              2098176

 dense_9 (Dense)             (None, 512)               524800

 dense_10 (Dense)            (None, 256)               131328

 dense_11 (Dense)            (None, 2)                 514

=================================================================
Total params: 24,557,602
Trainable params: 2,754,818
Non-trainable params: 21,802,784
_________________________________________________________________
```

### VGG16 Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688

 flatten (Flatten)           (None, 512)               0

 dense (Dense)               (None, 512)               262656

 dense_1 (Dense)             (None, 2)                 1026

=================================================================
Total params: 14,978,370
Trainable params: 263,682
Non-trainable params: 14,714,688
_________________________________________________________________
```

### VGG16 Dense Model

```_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688

 flatten (Flatten)           (None, 512)               0

 dense (Dense)               (None, 512)               262656

 dense_1 (Dense)             (None, 256)               131328

 dense_2 (Dense)             (None, 2)                 514

=================================================================
Total params: 15,109,186
Trainable params: 394,498
Non-trainable params: 14,714,688
_________________________________________________________________
```

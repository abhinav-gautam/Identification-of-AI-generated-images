# Model Performance Log

This log it to track trained model's performance and architectures. This will be helpful for tracking model's performance and improvements.

## Model Performance

| S. No. | Model Name                                | Training Dataset | Architecture      | Activation Function | Epoch | Optimizer | Learning Rate | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| ------ | ----------------------------------------- | ---------------- | ----------------- | ------------------- | ----- | --------- | ------------- | ------------- | ----------------- | --------------- | ------------------- |
| 1.     | basic_model_cifake                        | CIFAKE           | Basic Model       | relu                | 10    | Adam      | 0.0001        | 0.2816        | 0.8814            | 0.3470          | 0.8648              |
| 2.     | basic_tanh_model_cifake                   | CIFAKE           | Basic Model       | tanh                | 100   | Adam      | 0.0001        | 0.4088        | 0.8154            | 0.4040          | 0.8231              |
| 3.     | basic_model_ai_art                        | AI Art           | Basic Model       | relu                | 10    | Adam      | 0.0001        | 0.3819        | 0.8480            | 0.3862          | 0.8361              |
| 4.     | basic_model_merged                        | CIFAKE + AI Art  | Basic Model       | relu                | 10    | Adam      | 0.0001        | 0.3161        | 0.8689            | 0.4446          | 0.8171              |
| 5.     | advance_model_ai-art                      | AI Art           | Advance Model     | relu                | 10    | Adam      | 0.0001        | 0.3478        | 0.8611            | 0.4645          | 0.8080              |
| 6.     | advance_model_cifake                      | CIFAKE           | Advance Model     | relu                | 10    | Adam      | 0.0001        | 0.3291        | 0.8623            | 0.4437          | 0.8462              |
| 7.     | alexnet_model_cifake                      | CIFAKE           | AlexNet           | relu                | 10    | Adam      | 0.0001        | 0.6932        | 0.5003            | 0.6931          | 0.5000              |
| 8.     | lenet5_model_cifake                       | CIFAKE           | LeNet5            | relu                | 10    | Adam      | 0.0001        | 0.4181        | 0.8081            | 0.3678          | 0.8358              |
| 9.     | lenet5_dense_model_cifake                 | CIFAKE           | LeNet5 Dense      | relu                | 10    | Adam      | 0.0001        | 0.4565        | 0.7878            | 0.3578          | 0.8388              |
| 10.    | lenet5-100-epochs_model_cifake            | CIFAKE           | LeNet5 Dense      | relu                | 100   | Adam      | 0.0001        | 0.3698        | 0.8391            | 0.3793          | 0.8414              |
| 11.    | lenet5_model_ai_art                       | Ai Art           | LeNet5            | relu                | 10    | Adam      | 0.0001        | 0.4404        | 0.8253            | 0.4247          | 0.8253              |
| 12.    | resnet_model_cifake                       | CIFAKE           | ResNet50          | relu                | 10    | Adam      | 0.0001        | 0.4616        | 0.7867            | 0.8089          | 0.8100              |
| 13.    | resnet_model_ai_art                       | Ai Art           | ResNet50          | relu                | 10    | Adam      | 0.0001        | 0.4344        | 0.8210            | 1.0570          | 0.8274              |
| 14.    | inception-v3_model_ai-art                 | Ai Art           | InceptionV3       | relu                | 10    | Adam      | 0.0001        | 0.3762        | 0.8483            | 0.4263          | 0.8290              |
| 14.    | inception-v3_model_cifake                 | CIFAKE           | InceptionV3       | relu                | 10    | Adam      | 0.0001        | 0.4065        | 0.8147            | 0.3596          | 0.8418              |
| 14.    | inception-v3-dense_model_cifake           | CIFAKE           | InceptionV3 Dense | relu                | 10    | Adam      | 0.0001        | 0.3952        | 0.8211            | 0.3605          | 0.8422              |
| 14.    | inception-v3-dense-epochs100_model_cifake | CIFAKE           | InceptionV3 Dense | relu                | 100   | Adam      | 0.0001        | 0.3519        | 0.8474            | 0.3185          | 0.8649              |
| 14.    | inception-v3-dense-epochs100_model_cifake | CIFAKE           | InceptionV3 Dense | relu                | 100   | Adam      | 0.0001        | 0.3519        | 0.8474            | 0.3185          | 0.8649              |

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

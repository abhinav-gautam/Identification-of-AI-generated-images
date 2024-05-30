# Model Performance Log

This log it to track trained model's performance and architectures. This will be helpful for tracking model's performance and improvements.

## Model Architecture

### Basic Model

```python
Sequential([
    Convolution2D(100, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Convolution2D(100, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation="relu"),
    Dense(2, activation="softmax")
])
```

## Model Performance

| S. No. | Model Name         | Training Dataset         | Architecture | Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| ------ | ------------------ | ------------------------ | ------------ | ----- | ------------- | ----------------- | --------------- | ------------------- |
| 1.     | basic_model_cifake | CIFAKE                   | Basic Model  | 10    | 0.2816        | 0.8814            | 0.3470          | 0.8648              |
| 2.     | basic_model_ai_art | AI Art                   | Basic Model  | 10    | 0.3819        | 0.8480            | 0.3862          | 0.8361              |
| 3.     | basic_model_merged | CIFAKE + AI Art (Merged) | Basic Model  | 10    | 0.3161        | 0.8689            | 0.4446          | 0.8171              |

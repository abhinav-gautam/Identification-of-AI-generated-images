# Building model
if not model_history_loaded or force_model_train:
    # Create a new model
    inputs = Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(
        256,
        kernel_regularizer=L2(0.01),
        activity_regularizer=L1(0.01),
        bias_regularizer=L1(0.01),
        activation="relu",
    )(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    classifier = Model(inputs, outputs)

    # Compile the model
    classifier.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=["accuracy", Precision(), Recall()],
    )

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_dim, num_classes):
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    gesture_output = layers.Dense(num_classes, activation='softmax', name='gesture')(x)
    force_output = layers.Dense(1, activation='linear', name='force')(x)

    model = models.Model(inputs=inputs, outputs=[gesture_output, force_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            'gesture': 'sparse_categorical_crossentropy',
            'force': 'mse'
        },
        metrics={
            'gesture': 'accuracy',
            'force': 'mae'
        }
    )

    return model

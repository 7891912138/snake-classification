import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import prepare_data, create_generators

# 设置路径
csv_path = 'data/csv/train.csv'
base_image_path = 'data/train'
train_split_csv = 'data/train_split.csv'
val_split_csv = 'data/val_split.csv'

# 准备数据
train_df, val_df = prepare_data(csv_path, base_image_path, train_split_csv, val_split_csv)
train_generator, val_generator = create_generators(train_df, val_df)

# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid', name='venomous_output'),
    Dense(train_generator.num_classes, activation='softmax', name='class_output')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss={
        'class_output': 'sparse_categorical_crossentropy',
        'venomous_output': 'binary_crossentropy'
    },
    metrics={
        'class_output': 'accuracy',
        'venomous_output': 'accuracy'
    }
)

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# 保存模型
model.save('models/snake_classifier_model.h5')
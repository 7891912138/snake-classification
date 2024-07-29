import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 设置路径
test_csv_path = 'data/csv/test.csv'
base_image_path = 'data/test'

# 读取 CSV 文件
test_df = pd.read_csv(test_csv_path)

# 更新 CSV 文件中的图像路径
test_df['image_path'] = test_df.apply(lambda row: os.path.join(base_image_path, f"{row['UUID']}.jpg"), axis=1)

# 数据生成器
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col=None,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# 加载模型
model = load_model('models/snake_classifier_model.h5')

# 评估模型
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
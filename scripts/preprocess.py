import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_data(csv_path, base_image_path, train_split_csv, val_split_csv):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 创建 class_id 文件夹并移动图像
    for _, row in df.iterrows():
        class_id = row['class_id']
        uuid = row['UUID']
        image_filename = f"{uuid}.jpg"
        source_path = os.path.join(base_image_path, image_filename)
        class_dir = os.path.join(base_image_path, str(class_id))
        target_path = os.path.join(class_dir, image_filename)

        # 创建 class_id 文件夹（如果不存在）
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 移动图像
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)

    # 更新 CSV 文件中的图像路径
    df['image_path'] = df.apply(lambda row: os.path.join(base_image_path, str(row['class_id']), f"{row['UUID']}.jpg"),
                                axis=1)

    # 将数据集分为训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 保存分割后的数据
    train_df.to_csv(train_split_csv, index=False)
    val_df.to_csv(val_split_csv, index=False)

    return train_df, val_df


def create_generators(train_df, val_df):
    # 数据生成器
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col=['class_id', 'poisonous'],
        target_size=(150, 150),
        batch_size=32,
        class_mode='raw',  # 多输出模式
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col=['class_id', 'poisonous'],
        target_size=(150, 150),
        batch_size=32,
        class_mode='raw',
        shuffle=False
    )

    return train_generator, val_generator
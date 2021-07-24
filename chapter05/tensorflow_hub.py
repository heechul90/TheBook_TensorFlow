### 필요 라이브러리 호출
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import font_manager
font_fname = 'C:/Windows/Fonts/malgun.ttf'
font_family = font_manager.FontProperties(fname=font_fname).get_name()

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub


### resnet50 내려받기
model = tf.keras.Sequential([
          hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/4",
                         input_shape=(224,224,3),
                         trainable=False),
          tf.keras.layers.Dense(2, activation='softmax')
])


### 데이터 확장
BATCH_SIZE = 32
image_height = 100
image_width = 100
train_dir = 'chapter05/data/catanddog/train'
valid_dir = 'chapter05/data/catanddog/validation'

train = ImageDataGenerator(
                 rescale=1./255,
                 rotation_range=10,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.1
)

train_generator = train.flow_from_directory(train_dir,
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            shuffle=True,
                                            class_mode="categorical")

valid = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = valid.flow_from_directory(valid_dir,
                                            target_size=(image_height, image_width),
                                            color_mode="rgb",
                                            batch_size=BATCH_SIZE,
                                            seed=7,
                                            shuffle=True,
                                            class_mode="categorical")

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


### 라이브러리 및 데이터 호출
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img = load_img('chapter05/data/bird.jpg')
data = img_to_array(img)


### width_shift_range를 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(width_shift_range=[-200,200])
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### height_shift_range를 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(height_shift_range=0.5)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### flip을 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### rotation_range를 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(rotation_range=90)
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### brightness를 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(brightness_range=[0.3,1.2])
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### zoom을 이용한 이미지 증가
img_data = expand_dims(data, 0)
data_gen = ImageDataGenerator(zoom_range=[0.4,1.5])
data_iter = data_gen.flow(img_data, batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = data_iter.next()
    image = batch[0].astype('uint16')
    plt.imshow(image)
plt.show()


### 모델 훈련
history = model.fit(train_generator,
                    epochs=10,
                    validation_data=valid_generator,
                    verbose=2)


### 모델의 정확도를 시각적으로 표현
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, label="훈련 데이터셋")
plt.plot(epochs, val_accuracy, label="검증 데이터셋")
plt.legend()
plt.title('정확도')
plt.figure()

plt.plot(epochs, loss, label="훈련 데이터셋")
plt.plot(epochs, val_loss, label="검증 데이터셋")
plt.legend()
plt.title('오차')


### 이미지에 대한 예측 확인
class_names = ['cat', 'dog']
validation, label_batch = next(iter(valid_generator))
prediction_values = model.predict_classes(validation)

fig = plt.figure(figsize=(12,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(8):
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
    ax.imshow(validation[i,:], cmap=plt.cm.gray_r, interpolation='nearest')
    if prediction_values[i] == np.argmax(label_batch[i]):
        ax.text(3, 17, class_names[prediction_values[i]], color='yellow', fontsize=14)
    else:
        ax.text(3, 17, class_names[prediction_values[i]], color='red', fontsize=14)
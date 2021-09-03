### 필요한 라이브러리 호출
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 # 얼굴 인식, 물체 식별, 이미지 결합 등 작업이 가능한 오픈 소스 라이브러리

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

### VGG19 네트워크 생성
class VGG19(Sequential):
    def __init__(self, input_shape):

        super().__init__()
        self.add(Conv2D(64, kernel_size=(3,3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(Conv2D(64, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.add(Conv2D(128, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(128, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides= (2,2)))

        self.add(Conv2D(256, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(256, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(Conv2D(512, kernel_size=(3,3), padding='same',
                        activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(1000, activation='softmax'))

        self.compile(optimizer=tf.keras.optimizers.Adam(0.003),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])


### VGG19 모델 출력
model = VGG19(input_shape=(224,224,3))
model.summary()


### 사전 훈련된 VGG19 가중치 내려받기 및 클래스 정의
model.load_weights('chapter06/data/vgg19_weights_tf_dim_ordering_tf_kernels.h5') # 사전 훈련된 VGG19 모델의 가중치 내려받기
classes = {282: 'cat',
           681: 'notebook, notebook computer',
           970: 'alp'} # 검증용으로 사용될 클래스 세 개만 적용했으며, 전체 이미지에 대한 클래스는 “../chap6/data/”에 위치한 classes.txt 파일을 참고하세요.


### 이미지 호출 및 예측
image1 = cv2.imread('chapter06/data/labtop.jpg')
#image1 = cv2.imread('chapter06/data/starrynight.jpeg')
#image1 = cv2.imread('chapter06/data/cat.jpg')
image1 = cv2.resize(image1, (224,224))
plt.figure()
plt.imshow(image1)
image1 = image1[np.newaxis, :] # 차원 확장(행을 추가)
predicted_value = model.predict_classes(image1)
plt.title(classes[predicted_value[0]]) # 출력에 대한 title(제목) 지정

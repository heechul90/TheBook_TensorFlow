### 필요한 라이브러리 호출


### 아이덴티티 블록
def res_identity(x, filters):
    x_skip = x # 레지듀얼 블록을 추가하는 데 사용
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1,1), strides=(1,1), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x) # 첫 번째 블록

    x = Conv2D(f1, kernel_size=(3,3), strides=(1,1), padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x) # 두 번째 블록

    x = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x) # 세 번째 블록

    x = Add()([x, x_skip]) # 숏컷
    x = Activation(activations.relu)(x)
    return x


### 합성곱 블록
def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1,1), strides=(s,s), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x) # 첫 번째 블록

    x = Conv2D(f1, kernel_size=(3,3), strides=(1,1), padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x) # 두 번째 블록

    x = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='valid',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x) # 세 번째 블록

    x_skip = Conv2D(f2, kernel_size=(1,1), strides=(s,s), padding='valid',
                    kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip) # 숏컷

    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)
    return x
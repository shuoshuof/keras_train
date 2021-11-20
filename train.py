import keras
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Activation, Flatten, AveragePooling2D,MaxPooling2D,add,GlobalMaxPool2D,Dropout
from keras.models import Model, Sequential 
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 

import os 
import numpy as np 

x = np.load('./x.npy')
y = np.load('./y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 40)

np.save("test_x", x_test)
np.save("test_y", y_test)

x_train = x_train / 128.0 - 1
y_train = to_categorical(y_train)

x_test = x_test / 128.0 - 1
y_test = to_categorical(y_test)

pooling = MaxPool2D
def model():
    _in = Input(shape=(32,32,3))
    x = Conv2D(32, (3,3), padding='same')(_in)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = pooling((2,2))(x)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dense(8)(x)
    x = Activation("softmax")(x)
    return Model(_in, x)

    #残差网络
    #量化为tflite后会出现问题，openmv无法使用GlobalMaxPool2D，不量化能用
    # inputs =  Input(shape=(32, 32, 3), name='img')
    # h1 =   Conv2D(32, 3, activation='relu')(inputs)
    # h1 =   Conv2D(32, 3, activation='relu')(h1)
    # block1_out =   MaxPooling2D(3)(h1)
    #
    # h2 =   Conv2D(32, 3, activation='relu', padding='same')(block1_out)
    # h2 =   Conv2D(32, 3, activation='relu', padding='same')(h2)
    # block2_out =   add([h2, block1_out])
    #
    # # h3 =   Conv2D(64, 3, activation='relu', padding='same')(block2_out)
    # # h3 =   Conv2D(64, 3, activation='relu', padding='same')(h3)
    # # block3_out =   add([h3, block2_out])
    #
    # h4 =   Conv2D(32, 3, activation='relu')(block2_out)
    # h4 =   GlobalMaxPool2D()(h4)
    # h4 =   Dense(128, activation='relu')(h4)
    # h4 =   Dropout(0.5)(h4)
    # outputs =   Dense(8, activation='softmax')(h4)
    #
    #
    # return Model(inputs, outputs)




if __name__ == "__main__":
    if not (os.path.exists('./models')):
        os.mkdir("./models")
    model = model()
    model.summary()

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["acc"])
    early_stop = EarlyStopping(patience=20)
    reduce_lr = ReduceLROnPlateau(patience=15)
    save_weights = ModelCheckpoint("./models/model_{epoch:02d}_{val_acc:.4f}.h5", 
                                   save_best_only=True, monitor='val_acc')
    callbacks = [save_weights, reduce_lr]
    model.fit(x_train, y_train, epochs = 100, batch_size=32, 
              validation_data = (x_test, y_test), callbacks=callbacks)


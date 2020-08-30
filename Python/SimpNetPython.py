## Python version of SimpNet
# Used to train and export weights for the network 
# because it's fast

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras.backend import expand_dims
from keras import backend as K
from pandas import Series

# dimensions of our images.
img_width, img_height = 65, 65

train_data_dir = 'D:\\Storage\\Datasets\\Train\\Hololive Waifus'
validation_data_dir = 'D:\\Storage\\Datasets\\Test\\Hololive Waifus'
nb_train_samples = 20998
nb_validation_samples = 588
epochs = 5
batch_size = 100

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), strides=2, input_shape=input_shape))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), strides=2))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('elu'))
model.add(Dense(128))
model.add(Activation('elu'))
model.add(Dense(12))
model.add(Activation('softmax'))

if 1:
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        channel_shift_range=200.0,
        brightness_range=[-0.15, 0.15],
        fill_mode='nearest')
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights('SimpNet.h5')
    weights_list = model.get_weights()
    
    with open("WeightsCPP.txt", 'w') as f:
        f.write(Series(weights_list).to_json(orient='values'))
        f.close()
        
    with open("WeightsUNITY.txt", "w") as f:
        f.write("kern1:\n")
        for y in range(32):
            for x in range(27):
                i = (x // 9) % 3
                j = (x // 3) % 3
                k = x % 3
                l = y % 32
                f.write(repr(weights_list[0][i][j][k][l]) + " ")
            f.write("\n")
        f.write("bias1:\n")
        for y in range(32):
            f.write(repr(weights_list[1][y]) + " ")
        f.write("\nkern2:\n")
        for y in range(64):
            for x in range(288):
                i = (x // 96) % 3
                j = (x // 32) % 3
                k = x % 32
                l = y % 64
                f.write(repr(weights_list[2][i][j][k][l]) + " ")
            f.write("\n")
        f.write("bias2:\n")
        for y in range(64):
            f.write(repr(weights_list[3][y]) + " ")
        f.write("\nkern3:\n")
        for y in range(128):
            for x in range(576):
                i = (x // 192) % 3
                j = (x // 64) % 3
                k = x % 64
                l = y % 128
                f.write(repr(weights_list[4][i][j][k][l]) + " ")
            f.write("\n")
        f.write("bias3:\n")
        for y in range(128):
            f.write(repr(weights_list[5][y]) + " ")
        f.write("\nw1:\n")
        for y in range(128):
            for x in range(128):
                i = x % 128
                j = y % 128
                f.write(repr(weights_list[6][i][j]) + " ")
            f.write("\n")
        f.write("biasw1:\n")
        for y in range(128):
            f.write(repr(weights_list[7][y]) + " ")
        f.write("\nw2:\n")
        for y in range(128):
            for x in range(128):
                i = x % 128
                j = y % 128
                f.write(repr(weights_list[8][i][j]) + " ")
            f.write("\n")
        f.write("biasw2:\n")
        for y in range(128):
            f.write(repr(weights_list[9][y]) + " ")
        f.write("\nw3:\n")
        for y in range(12):
            for x in range(128):
                i = x % 128
                j = y % 12
                f.write(repr(weights_list[10][i][j]) + " ")
            f.write("\n")
        f.write("biasw3:\n")
        for y in range(12):
            f.write(repr(weights_list[11][y]) + " ")
        f.close()
    
else:
    model.load_weights('SimpNet.h5')
        
print(model.summary())

# # Predict
# img = load_img('D:\\Storage\\Datasets\\Test\\Fruits\\Apples\\10.jpg')
# # convert to numpy array
# img_np = img_to_array(img) / 255.
# new_image = expand_dims(img_np, 0)

# outputs = [K.function([model.input], [layer.output])([new_image, 1]) for layer in model.layers]
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

train_data_dir = 'D:\\Storage\\Datasets\\Train\\Fruits'
validation_data_dir = 'D:\\Storage\\Datasets\\Test\\Fruits'
nb_train_samples = 680
nb_validation_samples = 92
epochs = 30
batch_size = 30

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
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True)
    
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
    
    with open("Weights.txt", 'w') as f:
        f.write(Series(weights_list).to_json(orient='values'))
        f.close()
else:
    model.load_weights('SimpNet.h5')

print(model.summary())

# Predict
img = load_img('D:\\Storage\\Datasets\\Test\\Fruits\\Apples\\10.jpg')
# convert to numpy array
img_np = img_to_array(img) / 255.
new_image = expand_dims(img_np, 0)

outputs = [K.function([model.input], [layer.output])([new_image, 1]) for layer in model.layers]
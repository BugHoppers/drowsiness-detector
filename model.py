import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def bulidModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     padding='same', input_shape=(24, 24, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def preprocessData(dir):
    datagen = ImageDataGenerator(rescale=1./255)
    data_generator = datagen.flow_from_directory(dir,
                                                 target_size=(24, 24),
                                                 batch_size=32,
                                                 color_mode='grayscale',
                                                 class_mode='binary')

    return data_generator


def trainModel(model):
    datagen = preprocessData('./assets/dataset_EyeImages')
    model.fit_generator(datagen, steps_per_epoch=len(datagen)/32, epochs=50)
    model.save('eyeblink.hdf5')


def main():
    model = bulidModel()
    trainModel(model)


if __name__ == "__main__":
    main()

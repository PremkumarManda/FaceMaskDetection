import cv2
import tensorflow as tf
print('setup sucessful')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    validation_split = 0.2
)

#load trained data
train_data = train_datagen.flow_from_directory(
    'Dtaset/',
    target_size =(224,224),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training'
)

#load test data
valid_data = train_datagen.flow_from_directory(
    'Dtaset/',
    target_size=(224,224),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation'
)

print(train_data.class_indices)

#load mobilenet
base_model = MobileNetV2(
    weights='imagenet',
    include_top = False,
    input_shape=(224,224,3)
)

#base layer freezing 
for layer in base_model.layers:
    layer.trainable = False

#telling the features to the model from our data
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128,activation='relu')(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(inputs =base_model.input,outputs = x)


model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

early_stop = EarlyStopping(patience=1,restore_best_weights=True)

model.fit(train_data,
          validation_data=valid_data,
          epochs = 5,
          callbacks=early_stop)

model.save('Mask_mobilenetmodel.keras')

print('model traind sucessfully and saved ')


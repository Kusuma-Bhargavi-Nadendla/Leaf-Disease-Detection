from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#initialize nn

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#convert pooling features space to large feature vector for fully
#connected layer 
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense

from keras.layers import BatchNormalization
from keras.layers import Dropout

#basic cnn
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(128,128, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



train_datagen = ImageDataGenerator(rescale = None,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/naden/OneDrive/Desktop/MY PROJECTS/Leaf Disease Detection/dataset/train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
#print(test_datagen);
labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory('C:/Users/naden/OneDrive/Desktop/MY PROJECTS/Leaf Disease Detection/dataset/test',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

labels2 = (test_set.class_indices)
print(labels2)

model.fit(
    training_set,  # Training data generator
    steps_per_epoch=375,  # Total training samples // batch size
    epochs=10,  # Number of epochs
    validation_data=test_set,  # Validation data generator
    validation_steps=125  # Total validation samples // batch size
)

test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Part 3 - Making new predictions
# Save the model architecture to a JSON file
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)

# Save the weights to a `.weights.h5` file
model.save_weights("model1.weights.h5")
print("Saved model architecture and weights to disk")

import json
import random
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution1D, Flatten
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Load data and randomly shuffle data
with open('newData.json') as f:
    data = json.load(f)

random.shuffle(data)

texts = []
tags = []

for item in data:
    texts.append(item['text'])
    tags.append(item['tag'])

# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

train_posts = texts[:train_size]
train_tags = tags[:train_size]

test_posts = texts[train_size:]
test_tags = tags[train_size:]

# Confirm that we have a balanced dataset. I might have to change this code to enforce a balanced dataset.
train_tag_count = {'uart': 0, 'spi': 0, 'adc': 0, 'twi': 0, 'timer': 0, 'pwm': 0, 'i2c': 0, 'interrupt': 0}
test_tag_count = {'uart': 0, 'spi': 0, 'adc': 0, 'twi': 0, 'timer': 0, 'pwm': 0, 'i2c': 0, 'interrupt': 0}

for tag in train_tags:
    train_tag_count[tag] += 1

for tag in test_tags:
    test_tag_count[tag] += 1

print(train_tag_count)
print(test_tag_count)

max_words = 3000
tokenize = Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)

# Save dictionary
dictionary = tokenize.word_index

with open('dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Inspect the dimensions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# Can try tweaking the hyperparamaters and/or a more complex model.
batch_size = 32
epochs = 4

# Build the model

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model, when val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

# Evaluate the accuracy of our trained model. Achieves consistently 70% accuracy (upper limit).
# possible limiting factors: size of the data set, skew distribution of the number of occurrences of the
# different tags in the training data, quality of the data set, too simple model.
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(model.summary())


# Save model
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Here's how to generate a prediction on individual examples

text_labels = encoder.classes_

for i in range(3):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts[i])
    print('Actual label: ' + test_tags[i])
    print("Predicted label: " + predicted_label + "\n")
from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf

learning_rate = 0.0001
training_iters = 300  # steps
batch_size = 64

width = 20  # no. of features that will be extracted from our utterance
height = 80  # max length of utterance
classes = 10  # digits

batch = word_batch = speech_data.mfcc_batch_generator(batch_size)
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y #overfitting for now,cause slow gpu ;_;

# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='Adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 


model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training loops
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)

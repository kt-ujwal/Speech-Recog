import os
import tflearn
import speech_data as data
import tensorflow as tf

print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)
if tf.__version__ >= '0.12' and os.name == 'nt':
	print("sorry, tflearn is not ported to tensorflow 0.12 on windows yet!(?)")
	quit() 
speakers = data.get_speakers()
number_classes=len(speakers)
print("speakers",speakers)
batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
X,Y=next(batch)


# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192]) 
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)

demo_file = "1_ujwal_420.wav"
demo=data.load_wav_file(data.path + demo_file)
result=model.predict([demo])
result=data.one_hot_to_item(result,speakers)
print("predicted speaker for %s : result = %s "%(demo_file,result))

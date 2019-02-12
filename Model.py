import sys
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import numpy as np

file = sys.argv[1]
os.mkdir("ts")
os.chdir("ts")
os.mkdir("ts")
os.chdir("/home/nikita/Roman_Num")
shutil.move(file, './ts/ts')

json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

batch_size = 16
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("mnist_model.h5")
loaded_model.compile(loss="sparse_categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
datagen = ImageDataGenerator(rescale=1. / 255)


test_generator = datagen.flow_from_directory(
    './ts',
    target_size=(28, 28),
    batch_size=1,
    class_mode='sparse')

test_generator.reset()
predict = loaded_model.predict_generator(test_generator, steps = None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
key = np.argmax(predict)
dict = {0: 'Eight',1: 'Five',2 : 'Four',3 : 'One',4 : 'Seven',5 : 'Six',6 : 'Three',7 : 'Two'}
print(dict[key])
shutil.rmtree('./ts')


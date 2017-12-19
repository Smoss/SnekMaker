import png
import os
import pickle
import numpy
import itertools

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

labels = unpickle('batches.meta')

files = [x for x in os.listdir('.') if x.startswith('data')]
x = 0
for file in files:
	cifar_data = unpickle(file)
	pics = cifar_data[b'data']
	names = cifar_data[b'filenames']
	zipped = zip(pics, names)
	for pixels, name in zipped:
		image_3d = numpy.reshape(pixels, (3, 32,32)).transpose([1, 2, 0])
		f = open('Initial\\' + str(name,'utf-8'), 'wb')
		w = png.Writer(32, 32)
		print(str(name,'utf-8'))
		print(str(numpy.reshape(image_3d, (-1, 32*3))))
		w.write(f, numpy.reshape(image_3d, (-1, 32*3)))
		f.close()
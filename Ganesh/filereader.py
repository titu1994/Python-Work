#filereader
import os
from struct import unpack
import numpy as np

class Filereader :
	data = None
	path = None

	def __init__(self,path="."):
		self.path = path
		self.data = {
			"training": ["train-images.idx3-ubyte","train-labels.idx1-ubyte"],
			"testing": ["t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte"]
		}

	def getData(self,dataset="training",sample=1000):
		path_data = os.path.join(self.path,self.data[dataset][0])
		path_labels = os.path.join(self.path,self.data[dataset][1])

		with open(path_data,"rb") as d:
			magic_number,N,rows,columns = unpack(">IIII", d.read(16))
			data = np.fromfile(d,dtype=np.uint8).reshape(N,rows*columns)

		with open(path_labels,"rb") as l:
			magic_number, N = unpack(">II",l.read(8))
			label = np.fromfile(l,dtype=np.uint8)

		requiredLabels = np.zeros([N,10], dtype=np.uint8)

		for i in range(N):
			requiredLabels[i][label[i]] = 1

		return (data[:sample].astype(np.float32),requiredLabels[:sample].astype(np.float32),rows,columns)
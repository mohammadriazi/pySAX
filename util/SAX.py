import os
import numpy as np
from scipy.stats import norm
import Trie


class SAX(object):
	"""
	Implementaion of Symbolic Aggregate Approximation method.
	"""

	def __init__(self,w=4,a=4):
		self._w = w
		self._alpahbet_size = a
		self._betas = self._calculate_beta()


	def compute_PAA(self,C,type="symmetric"):
		"""
		Calculate Piece wise 
		Args:
			C:numpy array 
		Returns:
			C_bar:			
		"""
		n = len(C)
		w = self._w
		if n%w==0:
			C_bar = np.mean(C.reshape(w,n//w),axis=1)
		elif type=="symmetric":
			print("symmetric")
			k = n//w
			b = n%w
			weight = float(n/((w)*(k+b)))
			weight_mat = np.ones(shape=[w,k+b])*weight
			nums = np.zeros(shape=[w,k+b])
			start = 0
			for i in range(w):
				nums[i,:] = C[start:start+k+b]
				start += k
			C_bar = np.mean(nums*weight_mat,axis=1)
		return C_bar


	def normalize(self,x):
		eps = 1e-6
		mu = np.mean(x)
		std = np.std(x)
		if std < eps:
			return np.zeros(shape=x.shape)
		else:
			return (x-mu)/std

	
	def _calculate_beta(self):
		a = self._alpahbet_size
		split_points = []
		for i in range(1,a):
			split_points.append(norm.ppf(i/a))
		return split_points


	def convert_to_word(self,paa):
		word = ""
		for i in paa:
			offset = np.searchsorted(self._betas,i)
			word += chr(ord('a') + offset)
		return word


	def get_sax_word(self,C):
		paa = self.compute_PAA(self.normalize(C))
		return self.convert_to_word(paa)


	def compute_SAX(self,signal,window_size=6,hop_size=1,hop_fraction=None):
		if hop_fraction:
			hop_size = int(window_size*hop_fraction)

		start = 0
		sax = ""
		while start <= len(signal)-self._w:
			if start+window_size > len(signal):
				C = signal[start:]
			C = signal[start:start+window_size]
			sax += self.get_sax_word(C)
			start += hop_size

		return sax


	def get_beta_values(self):
		return self._betas


def main():
	sax = SAX(w=6,a=5)
	a = np.array([7,1,4,4,4,4])
	print(sax.compute_SAX(a))

	return
	# print(c)
	# print(sax.compute_PAA(c))
	# print(sax.normalize(c))
	c = bin(100)
	print(c)
	print(type(c))

if __name__ == '__main__':
	main()

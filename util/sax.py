import os
import numpy as np
from scipy.stats import norm
from trie import Trie


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
			C_bar:	PAA of C		
		"""
		n = len(C)
		w = self._w
		if n%w==0:
			C_bar = np.mean(C.reshape(w,n//w),axis=1)
		elif type=="symmetric":
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
		return (self.convert_to_word(paa),paa)


	def compute_SAX(self,signal,window_size=6,hop_size=1,hop_fraction=None):
		if hop_fraction:
			hop_size = int(window_size*hop_fraction)

		start = 0
		sax = ""
		while start <= len(signal)-self._w:
			
			if start+window_size > len(signal):
				C = signal[start:]
			else:
				C = signal[start:start+window_size]
			sax,paa = self.get_sax_word(C)
			yield (sax,paa,start)
			start += hop_size


	def MINDIST(self,q_hat,c_hat,n):
		"""
		Calculating the distance between q_hat and c_hat
		using MINDIST measure!
		"""
		w = len(q_hat)
		scaling_factor = float(n/w)
		sum_square = sum(self._MINDIST_cell(q_hat[i],c_hat[i])**2 for i in range(w))
		return np.sqrt(scaling_factor*sum_square)


	def _MINDIST_cell(r,c):
		betas = self._betas
		r_idx = ord(r)-ord('a')
		c_idx = ord(c)-ord('a')
		if abs(r_idx-c_idx) <= 1:
			return 0.0
		else:
			return betas[max(r_idx,c_idx)-1] - betas[min(r_idx,c_idx)]


	def get_beta_values(self):
		return self._betas


def main():
	sax = SAX(w=4,a=4)
	signal = np.array([1,4,3,6,8,-1,4,3])
	for s,_,start in sax.compute_SAX(signal,window_size=8,hop_size=2):
		print("sax_symbol= {} started from {}".format(s,start))



if __name__ == '__main__':
	main()

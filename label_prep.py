import os
from fse import CSplitIndexedList
from gensim.models import KeyedVectors
from fse.models import SIF
from fse.models.base_s2v import BaseSentence2VecModel

def split_func(string):
    return string.lower().split()

class SIF_embeddings:
	def __init__(self, model_path = None):

		if model_path[3:] == 'vec': # If it is a pre-trained word vector
			ft = KeyedVectors.load_word2vec_format(model_path)
			self.model = SIF(ft, components=10)

		elif model_path[-6:] == 'pickle': # Already trained sentence vector 
			self.model = BaseSentence2VecModel.load(model_path)

	def fit(self,data):
		inp = CSplitIndexedList(data, custom_split=split_func)
		self.model.train(inp)

	def __call__(self,transcript):
		return self.model.infer([(transcript.split(),0)])



import math
from collections import Counter
import numpy as np

class GLEU():

	def __init__(self,order=4) :
		self.order = order

	def corpus_gleu(self, candidates, references, sources):
		
		summed_stats = [0 for i in range(2*self.order+2)]

		for candidate, reference, source in zip(candidates, references, sources):
			stats = list(self.gleu_stats(candidate, reference, source))
			summed_stats = [sum(scores) for scores in zip(summed_stats, stats)]

		return self.gleu(summed_stats)

	def sentence_gleu(self, candidate, reference, source):
		return self.gleu(list(self.gleu_stats(candidate, reference, source)), True)

	def sentence_gleu_per_step(self, candidate, reference, source):

		reference_ngrams = self.get_all_ngrams(reference.split())
		source_ngrams = self.get_all_ngrams(source.split())
		s_ngram_diffs = []

		for i in range(self.order):
			s_ngrams = source_ngrams[i]
			r_ngrams = reference_ngrams[i]
			s_ngram_diffs.append(self.get_ngram_diff(s_ngrams,r_ngrams))
	
		candidate_words = candidate.split()
		per_step_gleu = []

		for t in range(len(candidate_words)):
			partial_candidate = " ".join(candidate_words[:t+1])
			per_step_gleu.append(self.gleu(list(self.gleu_stats(partial_candidate, reference, source, reference_ngrams, source_ngrams, s_ngram_diffs)), True))

		return per_step_gleu


	# Collect BLEU-relevant statistics for a single hypothesis/reference pair.
	# Return value is a generator yielding:
	# (c, r, numerator1, denominator1, ... numerator4, denominator4)
	# Summing the columns across calls to this function on an entire corpus
	# will produce a vector of statistics that can be used to compute GLEU
	def gleu_stats(self, candidate, reference, source, reference_ngrams = None, source_ngrams = None, s_ngram_diffs = None):
		candidate = candidate.split()
		reference = reference.split()
		source = source.split()

		candidate_ngrams = self.get_all_ngrams(candidate)

		if not reference_ngrams:
			reference_ngrams = self.get_all_ngrams(reference)
		if not source_ngrams:
			source_ngrams = self.get_all_ngrams(source)

		clen = len(candidate)
		rlen = len(reference)
		yield clen
		yield rlen

		for n in iter(range(1,self.order+1)):
			c_ngrams = candidate_ngrams[n-1]
			s_ngrams = source_ngrams[n-1]
			r_ngrams = reference_ngrams[n-1]

			if not s_ngram_diffs:
				s_ngram_diff = self.get_ngram_diff(s_ngrams,r_ngrams)
			else:
				s_ngram_diff = s_ngram_diffs[n-1]

			yield max([ sum( (c_ngrams & r_ngrams).values() ) - \
						sum( (c_ngrams & s_ngram_diff).values() ), 0 ])

			yield max([clen+1-n, 0])

	# Compute GLEU from collected statistics obtained by call(s) to gleu_stats
	def gleu(self,stats,smooth=False):
		
		if smooth: 
			# smooth stats:
			# cutoff irrelevant stats (if candidate is too short)
			# avoid log(0) error by setting 0 nominators to 0.1
			cutoff_idx = len(stats)
			for i in range(2, len(stats), 2):
				if stats[i] == 0: # if nominator is 0 smooth
					stats[i] = 0.1

				if stats[i+1] == 0: # if denominator is 0 cut off
					cutoff_idx = i 
					break

			stats = stats[:cutoff_idx]
		
		if len(list(filter(lambda x: x==0, stats))) > 0:
			return 0
		(c, r) = stats[:2]
				
		log_gleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4

		return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)

	def get_all_ngrams(self,sentence) :
		return [ self.get_ngram_counts(sentence,n) for n in range(1,self.order+1) ]

	def get_ngram_counts(self,sentence,n) :
		return Counter([tuple(sentence[i:i+n])
			for i in iter(range(len(sentence)+1-n))])

	def get_ngram_diff(self,a,b) :
		# returns ngrams in a but not in b
		diff = Counter(a)
		for k in (set(a) & set(b)) :
			del diff[k]
		return diff
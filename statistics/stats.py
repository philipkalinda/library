### statitics package

class Stats():
	"""Stats package with numerous statistical functions.
	"""
	def mean(x):
		return sum(x) / len(x)

	def median(x):
		if len(x) % 2 != 0:
			return sorted(list(x))[int((len(x)+1)/2)]
		else:
			return sum(sorted(list(x))[int((len(x)/2)):int((len(x)/2)+2)])/2

	def mode(x):
		mode_val = 0; mode_count = 0;
		for i in set(x):
			if x.count(i) > mode_count:
				mode_count=x.count(i)
				mode_val = i
		return mode_val

	def sum_squares(x):
		return sum([(i-Stats.mean(x))**2 for i in x])

	def variance(x, sample=False):
		if sample:
			return Stats.sum_squares(x)/(len(x)-1)
		else:	
			return Stats.sum_squares(x)/len(x)

	def standard_deviation(x, sample=False):
		if sample:
			return Stats.variance(x, sample=True) ** 0.5	
		else:
			return Stats.variance(x) ** 0.5

	def z_score(value, x):
		return (value-Stats.mean(x))/Stats.standard_deviation(x)

	def standard_error_of_mean(x,sample_size):
		"""x = list of means
		sample_size = size of samples to calculate means in list "x"
		"""
		return Stats.standard_deviation(x, sample=False) / sample_size

	def confidence_interval_95(x,sample_size):
		return tuple([(Stats.mean(x)-(1.96*Stats.standard_error_of_mean(x,sample_size))), (Stats.mean(x)+(1.96*Stats.standard_error_of_mean(x,sample_size)))])
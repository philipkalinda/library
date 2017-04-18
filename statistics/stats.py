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


	def one_sample_standard_error_of_mean(x):
		"""x = list of means
		"""
		return Stats.standard_deviation(x, sample=True) / (len(x)**0.5)


	def confidence_interval_95(x):
		"""x = list of values (sample)
		"""
		return tuple([(Stats.mean(x)-(1.96*Stats.one_sample_standard_error_of_mean(x))), (Stats.mean(x)+(1.96*Stats.one_sample_standard_error_of_mean(x)))])


	def confidence_interval_99(x):
		"""x = list of values (sample)
		"""
		return tuple([(Stats.mean(x)-(2.575*Stats.one_sample_standard_error_of_mean(x))), (Stats.mean(x)+(2.575*Stats.one_sample_standard_error_of_mean(x)))])


	def one_sample_t_score(x, mu):
		"""x = list of values (sample)
		mu = population mean
		"""
		return (Stats.mean(x) - mu) / Stats.one_sample_standard_error_of_mean(x)


	def cohens_d(x, mu):
		"""x = list of values (sample)
		mu = population mean
		Explination: How far two means are in standard units
		"""
		return (Stats.mean(x) - mu) / Stats.standard_deviation(x, sample=True)


	def one_sample_r_squared(x, mu):
		"""x = list of values (sample)
		mu = population mean
		Explination: Proportion of variance related to another variable
		"""
		return (Stats.one_sample_t_score(x, mu,sample_size))**2 / ((Stats.one_sample_t_score(x, mu)**2)+(len(x)-1))


	def two_sample_standard_error_of_mean(x,y):
		"""x = list of means
		y = list of means
		"""
		return ((Stats.variance(x,sample=True)/len(x))+(Stats.variance(y,sample=True)/len(y)))**0.5


	def two_sample_t_score(x,y,x_mu,y_mu):
		"""x = list of means
		y = list of means
		x_mu = population mean from which x came from
		y_mu = population mean from which y came from
		"""
		return ((Stats.mean(x)-Stats.mean(y)) - (x_mu-y_mu)) / Stats.two_sample_standard_error_of_mean(x,y)


	def f_score(X):
		"""X = list of lists of values
		xg = Stats.mean([Stats.mean(x) for x in X])
		ssb = sum([len(k)*((Stats.mean(k)-xg)**2) for k in X])
		ssw = sum([sum([(i-Stats.mean(k))**2 for i in k]) for k in X])
		dfb = len(X)-1
		dfw = sum([len(k) for k in X])-len(X)
		result = ((ssb)/(dfb)) / ((ssw)/(dfw))
		"""
		return ((sum([len(k)*((Stats.mean(k)-(Stats.mean([Stats.mean(x) for x in X])))**2) for k in X]))/(len(X)-1)) / ((sum([sum([(i-Stats.mean(k))**2 for i in k]) for k in X]))/(sum([len(k) for k in X])-len(X)))







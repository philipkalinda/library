### statitics package

class Stats():

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
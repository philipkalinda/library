class Vector(object):
	def __init__(self, coordinates):
		try:
			if not coordinates:
				raise ValueError
			self.coordinates = tuple(coordinates)
			self.dimension = len(coordinates)

		except ValueError:
			raise ValueError('The coordinates must be non-empty')

		except TypeError:
			raise TypeError('The coordinates must be iterable')

	def __str__(self):
		return 'Vector: {}'.format(self.coordinates)

	def __eq__(self,v):
		return self.coordinates == v.coordinates

	def plus(self, v):
		return Vector([x+y for x,y in zip(self.coordinates, v.coordinates)])

	def minus(self, v):
		return Vector([x-y for x,y in zip(self.coordinates, v.coordinates)])

	def times_scalar(self, scalar):
		new_coordinates = [scalar * x for x in self.coordinates]
		return Vector(new_coordinates)

	def magnitude(self):
		return (sum([x**2 for x in self.coordinates]))**0.5

	def direction(self):
		return self.times_scalar((1/self.magnitude()))
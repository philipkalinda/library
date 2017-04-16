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
		return Vector([scalar * x for x in self.coordinates])

	def magnitude(self):
		return (sum([x**2 for x in self.coordinates]))**0.5

	def normalize(self):
		try:
			return self.times_scalar((1/self.magnitude()))

		except ZeroDivisionError:
			raise Exception('Cannot mormalize the zero vector')

	def dot_product(self, v):
		return sum([x*y for x,y in zip(self.coordinates, v.coordinates)])

	def angle(self, v, in_degrees = False):
		from numpy import arccos
		try:
			if in_degrees:
				return 57.2957795*(arccos(self.normalize().dot_product(v.normalize())))
			else:
				return arccos(self.normalize().dot_product(v.normalize()))
		except Exception as e:
			if str(e) == self.CANNOT_NORMALIZE_SERO_VECTOR_MSG:
				raise Exception('Cannot compute an angle with the zero vector')
			else:
				raise e

	def is_zero(self, tolerance = 1e-10):
		return self.magnitude() < tolerance

	def is_parallel(self,v, tolerance=1e-10):
		return ( self.is_zero() or
			v.is_zero() or
			self.angle(v) == 0 or
			(self.angle(v) - 3.141592653589)<tolerance)
		return self.dot_product(v) == (self.magnitude()*v.magnitude())

	def is_orthoganal(self, v, tolerance = 1e-10):
		return self.dot_product(v) < tolerance



class dataSet(object):
	'''
	A flexible data set for training and validating grant year classification models
	'''

	def __init__(self, years):
		''' Return an empty data_set object with a variable number of time classes '''

		self.years = years

	def load_train(self, sample_size):
		''' Load training data, taking sample_size number of samples for each class'''

		for year in self.years:
			return(year)
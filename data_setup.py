import pandas as pd

class dataSet(object):
	'''
	A flexible data set for training and validating grant year classification models
	'''

	def __init__(self, years):
		''' Return an empty data_set object with a variable number of time classes '''

		self.years = years
		# the class_ints attribute stores the years as integers (0,1,2 etc). This is important for the one-hot encoding step
		self.class_ints = dict(zip(years, range(len(self.years))))
		self.train = {}
		self.test = {}

	def load_data(self, partition, year, n):
		''' Function for loading test/train data '''
		
		abstracts = []
		
		f = open('data/sample' + str(year) + '_' + partition + '.csv')
		for i in range(n):
			abstracts.append(f.readline())

		f.close()

		return abstracts

	def load_train(self, sample_size):
		''' Load training/test data, taking sample_size number of samples for each class '''

		for year in self.years:
			self.train[year] = self.load_data('train', year, sample_size)

	def load_test(self, sample_size):
		''' Load testing data, taking sample_size number of samples for each class '''

		for year in self.years:
			self.test[year] = self.load_data('test', year, sample_size)
	
	def dump_data(self, partition):
		''' Return all data as a two-column data frame '''
		
		if partition == "train":
			x = self.train
		elif partition =="test":
			x = self.test
		else:
			return("error: chose 'train' or 'test' as partition param")
		
		df = pd.DataFrame()
		for year, abstracts in x.items():
			df_year = pd.DataFrame({"record" : abstracts, "label" : [self.class_ints[year]] * len(abstracts)})
			df = df.append(df_year, ignore_index = True)

		# Shuffle rows
		df = df.sample(frac = 1).reset_index(drop = True)
		return(df)

from nltk import text, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


import re
import sys


def process_text(abstract_record, label):

	# Tokenize

	record = text(word_tokenize(abstract_record))

	# Remove stopwords

	record = [word for word in record if word not in stopwords.words('english')]

	# Stemming

	record = [PorterStemmer(word) for word in record]

	# Labelling

	if label == '1986':
		lab = [0,1]
	elif label == '2016':
		lab = [1,0]
	else:
		print("ERROR: label must be 1986 or 2016")
		sys.exit(1)

	record_lab = [record, lab]

	return record_lab
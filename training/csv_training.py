import csv
import re

import dedupe
from unidecode import unidecode

def preProcess(column):
	"""
	Do a little bit of data cleaning with the help of Unidecode and Regex.
	Things like casing, extra spaces, quotes and new lines can be ignored.
	"""
	column = unidecode(column)
	column = re.sub('  +', ' ', column)
	column = re.sub('\n', ' ', column)
	column = column.strip().strip('"').strip("'").lower().strip()
	# If data is missing, indicate that by setting the value to `None`
	if not column:
		column = None
	return column

def readData(filename):
	"""
	Read in our data from a CSV file and create a dictionary of records,
	where the key is a unique record ID and each value is dict
	"""

	data_d = {}
	with open(filename) as f:
		reader = csv.DictReader(f)
		for row in reader:
			clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
			row_id = int(row['Id'])
			data_d[row_id] = dict(clean_row)

	return data_d

def train_csv(input_file, fields):
	
	data_d = readData(input_file)
	
	deduper = dedupe.Dedupe(fields)

	deduper.prepare_training(data_d)

	print('starting active labeling...')

	dedupe.console_label(deduper)

	print('Done with labeling...')

	print('training...')
	deduper.train()

	settings_file = 'settings_file'
	training_file = 'training.json'

	with open(training_file, 'w') as tf:
		deduper.write_training(tf)

	with open(settings_file, 'wb') as sf:
		deduper.write_settings(sf)

if __name__ == '__main__':

	input_file = '' #Fill in the input file csv path
	
	# You need to make sure that the fields you mention here are in the csv file. Do not add id field here.
	fields = [] #Fill in the fields of interest as shown. 
	'''Ex: 
	fields = [
			{'field': 'Site name', 'type': 'String'},
			{'field': 'Address', 'type': 'String'},
			{'field': 'Zip', 'type': 'Exact', 'has missing': True},
			{'field': 'Phone', 'type': 'String', 'has missing': True}
	]'''
	
	train_csv(input_file,fields)
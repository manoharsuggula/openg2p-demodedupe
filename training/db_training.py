import psycopg2
import psycopg2.extras

import dedupe


if __name__ == '__main__':
	
	# ## Setup
	# Name the Settings file and training files here Examples shown.
	settings_file = 'pgsql_big_dedupe_example_settings'
	training_file = 'pgsql_big_dedupe_example_training.json'


	db_conf = {} #Fill the DB configuration here
	'''
	Ex: {	"NAME": "dedupe_example", #DB name
			"USER": "postgres",
			"PASSWORD": "xyz123",
			"HOST": "localhost",
			"PORT": "5432",
		}
	'''



	read_con = psycopg2.connect(database=db_conf['NAME'],
								user=db_conf['USER'],
								password=db_conf['PASSWORD'],
								host=db_conf['HOST'],
								cursor_factory=psycopg2.extras.RealDictCursor)

	write_con = psycopg2.connect(database=db_conf['NAME'],
								 user=db_conf['USER'],
								 password=db_conf['PASSWORD'],
								 host=db_conf['HOST'])
					   
	
	# write a query that will extract the data from the db for training. 
	# The query should have the fields of interest as columns along with id field
	# An example is shown below 
	# Try to keep the number of rows to be less than 3000. Training might crash otherwise.
	SELECT_QUERY = "SELECT donor_id, city, name, zip, state, address " \
				   "from processed_donors"

	# ## Training

	
	print("No settings file, creating deduper from scratch")

	# Define the fields dedupe will pay attention to
	#
	fields = [] #Fill in the fields of interest as shown
	# Make sure that the fields you mention here are in the query above.
	# Do not add id field here.
	

	# The address, city, and zip fields are often missing, so we'll
	# tell dedupe that, and we'll learn a model that take that into
	# account
	'''Ex:
	fields = [{'field': 'name', 'type': 'String'},
				{'field': 'address', 'type': 'String', 'has missing': True},
				{'field': 'city', 'type': 'ShortString', 'has missing': True},
				{'field': 'state', 'type': 'ShortString', 'has missing': True},
				{'field': 'zip', 'type': 'ShortString', 'has missing': True},
			]'''
	
	# Create a new deduper object and pass our data model to it.
	deduper = dedupe.Dedupe(fields, num_cores=4)
	print("Created dedupe object")

	# Named cursor runs server side with psycopg2
	with read_con.cursor('select query') as cur:
		cur.execute(SELECT_QUERY)
		temp_d = {i: row for i, row in enumerate(cur)}

	print(len(temp_d))

	deduper.prepare_training(temp_d)
		

	del temp_d

	# ## Active learning

	print('starting active labeling...')
	# Starts the training loop. Dedupe will find the next pair of records
	# it is least certain about and ask you to label them as duplicates
	# or not.

	# use 'y', 'n' and 'u' keys to flag duplicates
	# press 'f' when you are finished
	dedupe.console_label(deduper)
	# When finished, save our labeled, training pairs to disk
	with open(training_file, 'w') as tf:
		deduper.write_training(tf)

	# Notice our argument here
	#
	# `recall` is the proportion of true dupes pairs that the learned
	# rules must cover. You may want to reduce this if your are making
	# too many blocks and too many comparisons.
	deduper.train(recall=0.90)

	with open(settings_file, 'wb') as sf:
		deduper.write_settings(sf)

	# We can now remove some of the memory hogging objects we used
	# for training
	deduper.cleanup_training()

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
import re
import io
import itertools
import psycopg2
import psycopg2.extras
from typing import List, Dict
import csv
from unidecode import unidecode
import aiofiles
import uuid
import asyncio
from fastapi.responses import FileResponse


import numpy as np
import dedupe
import json

app = FastAPI()

csv_queue = {}

csv_input_directory = 'csv_input'
csv_output_directory = 'csv_output'

deduper = ""
pg_deduper = ""

read_con = ""
write_con = ""
id_field = ""
table = ""
fields = []

def load_file_on_startup(filename="configurations/settings_file"):
	try:
		with open(filename, 'rb') as f:
			deduper = dedupe.StaticDedupe(f)
		print("File loaded")
	except FileNotFoundError:
		print("File not found.")
	return deduper


# Fill the db_conf.json file with the db configuration
# The file should contain the following fields:
# 	 NAME: Name of the database
# 	 USER: Username of the database
# 	 PASSWORD: Password of the database
# 	 HOST: Host of the database
# 	 PORT: Port of the database
# 	 table: Table name to be used
# 	 id_field: Primary key of the table
# 	 fields: List of fields to be used for deduplication

def set_db_conf(db_conf):
	global read_con, write_con, id_field, table, fields
	read_con = psycopg2.connect(database=db_conf['NAME'],
								user=db_conf['USER'],
								password=db_conf['PASSWORD'],
								host=db_conf['HOST'],
								cursor_factory=psycopg2.extras.RealDictCursor)

	write_con = psycopg2.connect(database=db_conf['NAME'],
								user=db_conf['USER'],
								password=db_conf['PASSWORD'],
								host=db_conf['HOST'])
	id_field = db_conf["id_field"]
	table = db_conf["table"]
	fields = db_conf["fields"]

@app.on_event("startup")
async def startup_event():
	global pg_deduper, deduper
	pg_deduper = load_file_on_startup("configurations/db_settings")
	deduper = load_file_on_startup("configurations/settings_file")

	file_path = 'configurations/db_conf.json'
	with open(file_path, 'r') as file:
		db_conf = json.load(file)
	set_db_conf(db_conf)



########################## DB Dedupe ##########################
class Readable(object):

	def __init__(self, iterator):

		self.output = io.StringIO()
		self.writer = csv.writer(self.output)
		self.iterator = iterator

	def read(self, size):

		self.writer.writerows(itertools.islice(self.iterator, size))

		chunk = self.output.getvalue()
		self.output.seek(0)
		self.output.truncate(0)

		return chunk

def record_pairs(result_set):

	for i, row in enumerate(result_set):
		a_record_id, a_record, b_record_id, b_record = row
		record_a = (a_record_id, a_record)
		record_b = (b_record_id, b_record)

		yield record_a, record_b


def cluster_ids(clustered_dupes):

	for cluster, scores in clustered_dupes:
		cluster_id = cluster[0]
		for id, score in zip(cluster, scores):
			yield id, cluster_id, score

@app.post("/db_deduplicate/")
async def db_deduplicate(threshold:float):
	print(fields)
	SELECT_QUERY = "SELECT " + id_field + ", " + ', '.join(fields) + " from "+ table
	print(SELECT_QUERY)
	print('creating blocking_map database')
	with write_con:
		with write_con.cursor() as cur:
			cur.execute("DROP TABLE IF EXISTS blocking_map")
			cur.execute("CREATE TABLE blocking_map "
						"(block_key text, %s INTEGER)" % id_field)
			
	print('creating inverted index')

	print(pg_deduper.fingerprinter.index_fields.keys())
	for field in pg_deduper.fingerprinter.index_fields:
		with read_con.cursor('field_values') as cur:
			cur.execute("SELECT DISTINCT %s FROM %s" % (field, table))
			field_data = (row[field] for row in cur)
			pg_deduper.fingerprinter.index(field_data, field)

	print('writing blocking map')
	with read_con.cursor('select') as read_cur:
		read_cur.execute(SELECT_QUERY)

		full_data = ((row[id_field], row) for row in read_cur)
		b_data = pg_deduper.fingerprinter(full_data)

		with write_con:
			with write_con.cursor() as write_cur:
				write_cur.copy_expert('COPY blocking_map FROM STDIN WITH CSV',
									  Readable(b_data),
									  size=10000)
				
	pg_deduper.fingerprinter.reset_indices()
	with write_con:
		with write_con.cursor() as cur:
			cur.execute("CREATE UNIQUE INDEX ON blocking_map "
						"(block_key text_pattern_ops, %s)" % id_field)
			
	with write_con:
		with write_con.cursor() as cur:
			cur.execute("DROP TABLE IF EXISTS entity_map")

			print('creating entity_map database')
			cur.execute("CREATE TABLE entity_map "
						"(%s INTEGER, canon_id INTEGER, "
						" cluster_score FLOAT, PRIMARY KEY(%s))" % (id_field, id_field))

	query = f"""
		SELECT a.{id_field},
			row_to_json((SELECT d FROM (SELECT a.{', a.'.join(fields)}) d)),
			b.{id_field},
			row_to_json((SELECT d FROM (SELECT b.{', b.'.join(fields)}) d))
		FROM (SELECT DISTINCT l.{id_field} as east, r.{id_field} as west
			FROM blocking_map as l
			INNER JOIN blocking_map as r
			USING (block_key)
			WHERE l.{id_field} < r.{id_field}) ids
		INNER JOIN {table} a ON ids.east = a.{id_field}
		INNER JOIN {table} b ON ids.west = b.{id_field}
	"""
	# print(query)

	with read_con.cursor('pairs', cursor_factory=psycopg2.extensions.cursor) as read_cur:
		read_cur.execute(query)

		print('clustering...')
		clustered_dupes = pg_deduper.cluster(pg_deduper.score(record_pairs(read_cur)),
										  threshold=threshold)
		
		print('writing results')
		with write_con:
			with write_con.cursor() as write_cur:
				write_cur.copy_expert('COPY entity_map FROM STDIN WITH CSV',
									  Readable(cluster_ids(clustered_dupes)),
									  size=10000)

	with write_con:
		with write_con.cursor() as cur:
			cur.execute("CREATE INDEX head_index ON entity_map (canon_id)")

	return {"message": "success"}



########################## DB Dedupe End ##########################

########################## CSV Dedupe ##########################

def preProcess(column):
	column = unidecode(column)
	column = re.sub('  +', ' ', column)
	column = re.sub('\n', ' ', column)
	column = column.strip().strip('"').strip("'").lower().strip()
	
	if not column:
		column = None
	return column


async def process(threshold, txn_id):
	file_name = csv_input_directory+'/'+txn_id+'.csv'
	data_d = {}
	with open(file_name, 'r') as file:
		reader = csv.DictReader(file)
		for row in reader:
			clean_row = [(k, preProcess(v)) for (k, v) in row.items()]
			row_id = int(row['Id'])
			data_d[row_id] = dict(clean_row)

		# This field names will contain all the columns of csv not just fields of interest.
		field_names = []
		if reader.fieldnames:
			field_names = reader.fieldnames
			print(field_names)


	print('clustering...')
	clustered_dupes = deduper.partition(data_d, threshold)

	print('# duplicate sets', len(clustered_dupes))

	
	cluster_membership = {}
	for cluster_id, (records, scores) in enumerate(clustered_dupes):
		for record_id, score in zip(records, scores):
			cluster_membership[record_id] = {
				"Cluster ID": str(cluster_id),
				"confidence_score": str(score)
			}

	print(len(cluster_membership))

	with open(csv_output_directory+'/'+txn_id+'.csv', 'w') as csv_content:
		writer = csv.DictWriter(csv_content, fieldnames=['Cluster ID', 'confidence_score'] + field_names)
		writer.writeheader()

		for keys, values in cluster_membership.items():
			values.update(data_d[keys])
			writer.writerow(values)

	csv_queue[txn_id] = "completed"


@app.get("/csv_deduplicate_download/{txn_id}")
async def csv_deduplicate_download(txn_id):
	if(csv_queue[txn_id] == "completed"):
		return FileResponse(csv_output_directory+'/'+txn_id+'.csv', media_type='text/csv', filename=txn_id+'.csv')
	else:
		return {"status": "processing"}

@app.get("/csv_deduplicate_status/{txn_id}")
async def csv_deduplicate_status(txn_id):
	return {"status": csv_queue[txn_id]}

@app.post("/csv_deduplicate/")
async def csv_deduplicate(threshold:float, in_file: UploadFile):
	txn_id = str(uuid.uuid4())
	async with aiofiles.open(csv_input_directory+'/'+txn_id+'.csv', 'wb') as out_file:
		content = await in_file.read()  # async read
		await out_file.write(content)

	asyncio.create_task(process(threshold, txn_id))
	csv_queue[txn_id] = "processing"
	return {"txn_id": txn_id}

########################## CSV Dedupe End ##########################

########################## JSON Dedupe ##########################

class DictInput(BaseModel):
	data: dict

@app.post("/json_deduplicate/")
async def json_deduplicate(threshold:float,input_data: DictInput):
	try:
		
		input_json = input_data.data
		if isinstance(input_json, dict):
			data_d = input_json
		else:
			data_d = json.loads(input_json)

		print('clustering...')
		clustered_dupes = deduper.partition(data_d, threshold)

		print('# duplicate sets', len(clustered_dupes))

		
		cluster_membership = {}
		for cluster_id, (records, scores) in enumerate(clustered_dupes):
			for record_id, score in zip(records, scores):
				cluster_membership[record_id] = {
					"Cluster ID": str(cluster_id),
					"confidence_score": str(score)
				}


		clusters = {}

		for record_id, cluster_info in cluster_membership.items():
			cluster_id = cluster_info["Cluster ID"]
			
			if cluster_id not in clusters:
				clusters[cluster_id] = []

			clusters[cluster_id].append(record_id)

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))	

	return clusters

########################## JSON Dedupe End ##########################


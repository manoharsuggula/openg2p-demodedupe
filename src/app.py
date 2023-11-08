from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import re
import time
import itertools
import psycopg2
import psycopg2.extras
from typing import List, Dict
from io import StringIO
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

def load_file_on_startup(filename="settings_file"):
	global deduper
	try:
		with open(filename, 'rb') as f:
			deduper = dedupe.StaticDedupe(f)
		print("File loaded")
	except FileNotFoundError:
		print("File not found.")

@app.on_event("startup")
async def startup_event():
	load_file_on_startup()

class DictInput(BaseModel):
	data: dict

class ListInput(BaseModel):
	data: List[dict]

class TrainingData(BaseModel):
   settings_file: str
   training_file: str

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
	return FileResponse(csv_output_directory+'/'+txn_id+'.csv', media_type='text/csv', filename=txn_id+'.csv')

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

		# print(clusters)

		# output_dict = {}

		# for key in set(data_d) | set(cluster_membership):
		# 	output_dict[key] = {}

		# 	for sub_dict in [data_d, cluster_membership]:
		# 		if key in sub_dict:
		# 			output_dict[key].update(sub_dict[key])

		# print(output_dict)
		# output_dict = dict(itertools.islice(output_dict.items(), 100))

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))	

	return clusters

# @app.post("/train-dedupe/")
# async def train_deduper(input_data: DictInput, fields_data : ListInput):
# 	try:
# 		print("Training deduper...")
# 		input_json = input_data.data
# 		if isinstance(input_json, dict):
# 			data_d = input_json
# 		else:
# 			data_d = json.loads(input_json)

# 		f = fields_data.data
# 		fields = []

# 		for i in range(len(f)):
# 			if isinstance(f[i], dict):
# 				fields.append(f[i])
# 			else:
# 				fields.append(json.loads(f[i]))

		
# 		deduper = dedupe.Dedupe(fields)

# 		deduper.prepare_training(data_d)

# 		print('starting active labeling...')

# 		dedupe.console_label(deduper)

# 		print('Done with labeling...')

# 		print('training...')
# 		deduper.train()

# 		settings_file = 'settings_file'
# 		training_file = 'training.json'

# 		with open(training_file, 'w') as tf:
# 			deduper.write_training(tf)

# 		with open(settings_file, 'wb') as sf:
# 			deduper.write_settings(sf)

# 	except Exception as e:
# 		raise HTTPException(status_code=500, detail=str(e))

# 	return {"message": "Deduper trained successfully"}
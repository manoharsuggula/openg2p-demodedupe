from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import re
import time
import itertools
import psycopg2
import psycopg2.extras
from typing import List, Dict

import numpy as np
import dedupe
import json

app = FastAPI()


class DictInput(BaseModel):
    data: dict

class ListInput(BaseModel):
    data: List[dict]

class TrainingData(BaseModel):
   settings_file: str
   training_file: str


@app.post("/train-dedupe/")
async def train_deduper(input_data: DictInput, fields_data : ListInput):
	try:
		print("Training deduper...")
		input_json = input_data.data
		if isinstance(input_json, dict):
			data_d = input_json
		else:
			data_d = json.loads(input_json)

		f = fields_data.data
		fields = []

		for i in range(len(f)):
			if isinstance(f[i], dict):
				fields.append(f[i])
			else:
				fields.append(json.loads(f[i]))

		
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

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

	return {"message": "Deduper trained successfully"}

@app.post("/deduplicate/")
async def deduplicate(input_data: DictInput, files: TrainingData):
	try:
		settings_file = files.settings_file
		training_file = files.training_file

		input_json = input_data.data
		if isinstance(input_json, dict):
			data_d = input_json
		else:
			data_d = json.loads(input_json)
		
		with open(settings_file, 'rb') as f:
			deduper = dedupe.StaticDedupe(f)
		
			
		print('clustering...')
		clustered_dupes = deduper.partition(data_d, 0.5)

		print('# duplicate sets', len(clustered_dupes))

		
		cluster_membership = {}
		for cluster_id, (records, scores) in enumerate(clustered_dupes):
			for record_id, score in zip(records, scores):
				cluster_membership[record_id] = {
					"Cluster ID": str(cluster_id),
					"confidence_score": str(score)
				}

		output_dict = {}

		for key in set(data_d) | set(cluster_membership):
			output_dict[key] = {}

			for sub_dict in [data_d, cluster_membership]:
				if key in sub_dict:
					output_dict[key].update(sub_dict[key])

		print(output_dict)
		output_dict = dict(itertools.islice(output_dict.items(), 100))

	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))	

	return output_dict

'''
Files

'''
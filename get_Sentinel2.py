import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage

PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = 'treeai_swiss'
EXP_SCALE = 10
MAX_PIXELS = 1e13


# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)


import os
import json

# Contains configuration settings for the JSON Distributor
if os.path.exists('./local.settings.json'):
    
    f = open('local.settings.json')
    settings = json.load(f)
    secrets = settings['Values']

    os.environ['STORAGE_CONNECTION_STRING'] = secrets['AzureWebJobsStorage']

class ConfigSettings(object):
    STORAGE_CONNECTION_STRING = os.environ['STORAGE_CONNECTION_STRING']

    RAW_IMAGES_CONTAINERNAME = 'rawimages'
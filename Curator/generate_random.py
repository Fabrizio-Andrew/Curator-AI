import sys
import json
from flask import Flask, jsonify
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import time
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
from scipy.stats import truncnorm
import random
import base64
from io import BytesIO
import PIL.Image
import datetime
import logging
import uuid
from .config import ConfigSettings

import azure.functions as func


module_path = 'https://tfhub.dev/deepmind/biggan-256/2'
rand_seed = 123
truncation = 0.5

tf.reset_default_graph()
tf.disable_eager_execution()
logging.info(f'Loading BigGAN module from: {module_path}')
module = hub.Module(module_path)
logging.info('BigGAN module loaded')
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
        for k, v in module.get_input_info_dict().items()}
output = module(inputs)

input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']
random_state = np.random.RandomState(rand_seed)
dim_z = input_z.shape.as_list()[1]
vocab_size = input_y.shape.as_list()[1]

initializer = tf.global_variables_initializer()

sess = tf.Session()
sess.run(initializer)



def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    logging.info('Python timer trigger function ran at %s', utc_timestamp)    

    app = Flask(__name__, static_url_path='') #, static_folder='public', )

    # Create the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(ConfigSettings.STORAGE_CONNECTION_STRING)

    # Create the output container (if it doesn't already exist)
    output_container = ContainerClient.from_connection_string(ConfigSettings.STORAGE_CONNECTION_STRING, ConfigSettings.RAW_IMAGES_CONTAINERNAME)
    try:
        container_properties = output_container.get_container_properties()
    except Exception as e:
        output_container = blob_service_client.create_container(ConfigSettings.RAW_IMAGES_CONTAINERNAME)

    try:
        num = 2
        print('Random', num)
        t = time.time()
        imgs, vectors, labels = create_random_images(num, max_classes=3)
        print('Finished in', time.time()-t)
        #with app.app_context():
            #img_list = jsonify([
            #    [ encode_img(arr) for arr in imgs ],
            #    vectors.tolist(),
            #    labels.tolist()
            #])
        error_count = 0

        for arr in imgs:
            data = encode_img(arr)

        # create file with responses
        #for data in img_list.data[0]:

            if data.startswith('data:image/jpeg;base64,'):
                
                # Format the data string & create filename
                data = data[23:]
                converted_data = base64.b64decode(data)
                #file_body = bytes("b'" + str(data) + "'")
                filename = str(uuid.uuid1()) + '.jpg'
                # upload the file
                UploadBlob(filename, converted_data, ConfigSettings.RAW_IMAGES_CONTAINERNAME, blob_service_client)

            else:
                error_count+=1
                
        if error_count > 1:
            logging.error(f'Unexpected data format. {error_count} object(s) did not start with "data:image/jpeg;base64,"')

    except Exception as e:
        logging.error(e)
        return '', 500


def sample(vectors, labels, batch_size=10):
    num = vectors.shape[0]
    ims = []
    for batch_start in range(0, num, batch_size):
        s = slice(batch_start, min(num, batch_start + batch_size))
        feed_dict = {input_z: vectors[s], input_y: labels[s], input_trunc: truncation}
        ims.append(sess.run(output, feed_dict=feed_dict))
    ims = np.concatenate(ims, axis=0)
    assert ims.shape[0] == num
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    return ims


def create_random_images(num_images, max_classes):
    vectors = truncated_z_sample(num_images)
    labels = create_labels(num_images, max_classes)
    ims = sample(vectors, labels)

    return ims, vectors, labels


def truncated_z_sample(batch_size):
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=random_state)
    return truncation * values


def create_labels(num, max_classes):
    label = np.zeros((num, vocab_size))
    for i in range(len(label)):
        for _ in range(random.randint(1, max_classes)):
            j = random.randint(0, vocab_size-1)
            label[i, j] = random.random()
        label[i] /= label[i].sum()
    return label


def encode_img(arr):
    # Encode uint8 values array into base64 string for sending.
    image = PIL.Image.fromarray(arr)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=90)
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = 'data:image/jpeg;base64,'+img_bytes.decode('ascii')
    return img_str


def UploadBlob(filename, file_body, cont, bsc):
    """
    Uploads a specified file to a specified azure storage blob container.
    """

    # Write data to the file
    file = open(filename, 'wb')
    file.write(file_body)
    file.close()

    # Create the blob client
    blob_client = bsc.get_blob_client(container=cont, blob=filename)

    # Upload the file
    with open(filename, 'rb') as outfile:
        blob_client.upload_blob(outfile, overwrite=True)
import requests
import base64
import datetime
import argparse
import json
import wave

# decidere se passare i nomi dei file come argomeni o 
# sciverli di default
# o qlc di più complesso


parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', type=str)
args = parser.parse_args()

# add the MLP and CNN tflite models to the model registry

for filename in args.models:
	model_file = filename.split('/')[-1]
	model_name = model_file.split('.')[0]

	model = open(filename, 'rb').read()

	# create base64 strings
	model_base64b = base64.b64encode(model)
	model_string = model_base64b.decode()

	url = 'http://192.168.1.216:8080/add'

	payload = {'model': model_string, 'name': model_name}
	payload_json = json.dumps(payload)

	r = requests.put(url, data=payload_json)

	if r.status_code == 200:
		print(f"{model_name} saved successfully!")

	else:
		print("Error: ", r.status_code)



# list the models stored in the Model Registry and verify that their number is 2
url_list = 'http://192.168.1.216:8080/list'
r2 = requests.get(url_list)

if r2.status_code == 200:
	body = r2.json()
	
	if len(body['models']) != 2:
		print("Error! The folder does not contain two models!")
	
	print("Models stored in the 'models' folder:")
	for model in body['models']:
		print(model)
else:
	print("Error: ", r2.status_code)

# Invoke the registry to measure and predict temperature and humidity with the 
# CNN model. Set tthres=0.1 and tthres=0.2

# /predict
tthres = 0.1
hthres = 0.2
model = model_name # takes the last model inserted, but can be customized
url_pred = f'http://192.168.1.216:8080/predict/get?tthres={tthres}&hthres={hthres}&model={model}'
r3 = requests.get(url_pred)

if r3.status_code == 200:
	print("Prediction done.\n")
else:
	print("Error: ", r3.status_code)






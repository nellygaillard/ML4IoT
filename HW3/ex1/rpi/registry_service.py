import cherrypy
import json
import base64
import os
from DoSomething import DoSomething
from datetime import datetime
import time
import numpy as np
import adafruit_dht
import tensorflow as tf
import tensorflow.lite as tflite
from board import D4

@cherrypy.expose
class Add(object):

	def GET(self, *path, **query):
		pass

	def POST(self, *path, **query):
		pass

	def PUT(self, *path, **query):

		# we expect fixed length for operand and empty query
		if len(query) > 0:
			raise cherrypy.HTTPError(400, 'Wrong query')
		if len(path) > 0:
			raise cherrypy.HTTPError(400, "Wrong path")

		body = cherrypy.request.body.read()
		body = json.loads(body)

		model = body.get('model')
		if model is None:
			raise cherrypy.HTTPError(400, "Model is missing.")

		name = body.get('name')
		if name is None:
			raise cherrypy.HTTPError(400, "Name of the model is missing.")

		# transform from b64bytes string to bytes
		model_bytes = base64.b64decode(model)
		if not os.path.exists("./models/"):
			os.mkdir("models/")

		# store the model in local subfolder called models
		path = "./models/" + name + '.tflite'
		with open(path, 'wb') as f:
			f.write(model_bytes)
		f.close()


	def DELETE(self, *path, **query):
		pass


@cherrypy.expose
class List(object):

	def GET(self, *path, **query):
		if len(path) > 0:
			raise cherrypy.HTTPError(400, "Wrong path")
		if len(query) > 0:
			raise cherrypy.HTTPError(400, "Wrong query")

		models = []
		for filename in os.listdir('./models'):
			model_name = filename.split('.')[0]
			if len(model_name) > 0:
					models.append(model_name)

		body = {'models': models, 'path':path}
		body_json = json.dumps(body)

		return body_json

		
	def POST(self, *path, **query):
		pass

	def PUT(self, *path, **query):
		pass

	def DELETE(self, *path, **query):
		pass

@cherrypy.expose
class Predict(object):

	def __init__(self):
		self.dht_device = adafruit_dht.DHT11(D4)
		self.publisher = DoSomething("publisher TH")

	def GET(self, *path, **query):
		# check parameters
		if len(path) != 1:
			raise cherrypy.HTTPError(400, "Wrong path")

		if len(query) != 3:
			raise cherrypy.HTTPError(400, "Wrong query")
		
		self.publisher.run()

		# answer to the predict request
		tthres = float(query.get('tthres'))
		hthres = float(query.get('hthres'))
		model_name = query.get('model')

		if (model_name + '.tflite') not in os.listdir('./models'):
			raise cherrypy.HTTPError(400, "Wrong model name. It does not exists.")

		# load the tflite model
		interpreter = tflite.Interpreter(model_path=f"/home/pi/WORK_DIR/HW3/models/{model_name}.tflite")
		interpreter.allocate_tensors() 
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		# collect data: the model takes as input a (6,2) array
		buffer = np.zeros([1,6,2], dtype=np.float32)
		expected = np.zeros(2, dtype=np.float32)

		# we need to normalize the data to make the model work
		MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
		STD = np.array([8.654227, 16.557089], dtype=np.float32)

		# read first window
		for t in range(6):
			temperature = self.dht_device.temperature
			humidity = self.dht_device.humidity
			buffer[0,t,0] = np.float32(temperature)
			buffer[0,t,1] = np.float32(humidity)
			time.sleep(1)
		
		# normalization
		new_buffer = (buffer - MEAN) / STD

		while True:
			# read real values
			expected[0] = np.float32(self.dht_device.temperature)
			expected[1] = np.float32(self.dht_device.temperature)
			
			buffer = new_buffer
			
			# predicted values
			interpreter.set_tensor(input_details[0]['index'], buffer)
			interpreter.invoke()
			output = interpreter.get_tensor(output_details[0]['index'])
			output = np.squeeze(output)
		 
			# compute error 
			err_t = np.abs(output[0] - expected[0])
			err_h = np.abs(output[1] - expected[1])

			# if greather than threshold, send an alert
			now = datetime.now()
			timestamp = str((now.timestamp()))

			# send the alert
			if err_t > tthres:
				alert = {'bn':"Temperature",
						'bt':timestamp,
						'e':[
							{'n':'Predicted', 'u':'Cel', 't':0, 'v':str(round(output[0], 2))},
							{'n':'Real', 'u':'Cel', 't':0, 'v':str(expected[0])}
							]
						}							
				alert_json = json.dumps(alert)
				self.publisher.myMqttClient.myPublish("/282613/TH_alert", alert_json)

			if err_h > hthres:
				alert = {'bn':"Humidity",
						'bt':timestamp,
						'e':[
							{'n':'Predicted', 'u':'%', 't':0, 'v':str(round(output[1], 2))},
							{'n':'Real', 'u':'%', 't':0, 'v':str(expected[1])}
							]
						}
				alert_json = json.dumps(alert)
				self.publisher.myMqttClient.myPublish("/282613/TH_alert", alert_json)
				
			# prepare for new prediction
			new_val = np.reshape(expected, (1,1,2))
			new_val = (new_val - MEAN) / STD
			mini_buffer = buffer[:, 1: , :]
			new_buffer = np.hstack((mini_buffer, new_val))
			time.sleep(1)

	def POST(self, *path, **query):
		pass

	def PUT(self, *path, **query):
		pass

	def DELETE(self, *path, **query):
		pass


if __name__ == '__main__':

	conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
				'tools.sessions.on': True}}

	cherrypy.tree.mount(Add(), '/add', conf)
	cherrypy.tree.mount(List(), '/list', conf)
	cherrypy.tree.mount(Predict(), '/predict', conf)

	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})

	cherrypy.engine.start()
	cherrypy.engine.block()











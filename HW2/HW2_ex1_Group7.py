import pandas as pd
import numpy as np
import tensorflow as tf
import os
import argparse
from tensorflow import keras
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
import zlib


class WindowGenerator:
	def __init__(self, input_width, mean, std, label_options, out_steps):
		self.label_options = label_options
		self.input_width = input_width
		self.out_steps = out_steps
		# when we train we work on batches of data (4th dimension of every tensor)
		self.mean = tf.reshape(tf.convert_to_tensor(mean), [1,1,2])
		self.std = tf.reshape(tf.convert_to_tensor(std), [1,1,2])

	def split_window(self, features):
		"""
		Mehtod to generate the windows.
		Features is the data we want to split.
		Features contain 7 values (both for temperature and humidity).
		We are going to create a window of 6 plus the last element as label.
		"""
		inputs = features[:,:-self.out_steps,:]	# (batches, points, features)

		# shape of labels is (Batches, ). I need to reshape to (Batches, 1)
		if self.label_options < 2:
			labels = features[:, -self.out_steps:, self.label_options]
			labels = tf.expand_dims(labels, -1)
			num_labels = 1
		else:
			labels = features[:, -self.out_steps:, :]
			num_labels = 2

		# the batch size can be any
		inputs.set_shape([None, self.input_width, 2])
		labels.set_shape([None, self.out_steps, num_labels])

		return inputs, labels

	def preprocess(self, features):
		inputs, labels = self.split_window(features)
		inputs = self.normalize(inputs)

		return inputs, labels

	def normalize(self, features):
		features = (features - self.mean) / (self.std + 1.e-6)
		return features

	def make_dataset(self, data, train):
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
			data=data,
			targets=None,
			sequence_length=self.input_width + self.out_steps,
			sequence_stride=1,
			batch_size=32)
		ds = ds.map(self.preprocess)
		ds = ds.cache()

		if train is True:
			# we specify the size of the buffer used by tensorflow
			# the shuffle will be done at each epoch
			ds = ds.shuffle(100, reshuffle_each_iteration=True)	
		
		return ds


class My_MeanAbsoluteError(tf.keras.metrics.Metric):
    
    def __init__(self, name="MyMAE"):
        super().__init__(name=name)
        self.acc_th = self.add_weight(name="accumulator", shape=[2], initializer='zeros')
        self.counter = self.add_weight(name="counter", initializer='zeros')

        
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates metric statistics.
        In the case of multiple outputs, y_pred and y_true are vectors.
        """
        delta = tf.cast(tf.math.abs(y_pred - y_true), dtype=tf.float32)
        self.acc_th.assign_add(tf.reduce_mean(delta, axis=[0,1]))    # compute the mean on batch size and window
        self.counter.assign_add(1)


    def result(self):
        """
        Compute and returns the scalar metric value tensor or 
        a dictionary of tensors
        """
        return tf.math.divide_no_nan(self.acc_th, self.counter)

    def reset_states(self):
        """
        It resets the variable states between epochs.
        """
        self.acc_th.assign(tf.zeros_like(self.acc_th))
        self.counter.assign(tf.zeros_like(self.counter))
    

class MyModel():
	def __init__(self, input_width, label_options, X_train, version, out_steps, batch_size=32, alpha=1):

		if version=='b':
			model_MLP = keras.Sequential([
			    keras.layers.Flatten(input_shape=(input_width, label_options), name='first_flatten'),
			    keras.layers.Dense(int(alpha*16), activation='relu', name='first_dense'),
			    keras.layers.Dense(int(alpha*32), activation='relu', name='second_dense'),
			    keras.layers.Dense(out_steps*label_options, name='final_dense'), 
			    keras.layers.Reshape((out_steps, label_options), name='final_reshape')
			])
		elif version=='a':
				model_MLP = keras.Sequential([
			    keras.layers.Flatten(input_shape=(input_width, label_options), name='first_flatten'),
			    keras.layers.Dense(int(alpha*32), activation='relu', name='first_dense'),
			    keras.layers.Dense(out_steps*label_options, name='final_dense'), 
			    keras.layers.Reshape((out_steps, label_options), name='final_reshape')
			])

		self.version = version
		self.model = model_MLP
		self.X_train = X_train
		self.batch_size = batch_size
		self.pruning = False
		self.clustering = False
		self.quantization = False
		self.saved_model_dir = None
		self.epochs = None
		


	def train_base_model(self, optimizer, loss_function, epochs=10):

		self.epochs=epochs

		# compile the model
		self.model.compile(optimizer=optimizer, 
							loss=loss_function,
							metrics=My_MeanAbsoluteError())

		self.model.summary()

		# define the callbacks
		tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True)
		lr_callback = tf.keras.callbacks.LearningRateScheduler(self.my_scheduler, verbose=1)
		es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience = 4, verbose = 0, mode='auto', restore_best_weights=True)
		callbacks = [lr_callback, es_callback]
		
		if self.pruning:
			pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
			pruning_log = tfmot.sparsity.keras.PruningSummaries(log_dir='logs_pruning')
			callbacks.append(pruning_callback)

		# training
		history = self.model.fit(self.X_train, 
					batch_size=self.batch_size, 
                                        epochs=epochs,
					callbacks=callbacks)
		return history


	def my_mae_t(self, y_true, y_pred):
		delta = tf.cast(tf.math.abs(y_pred - y_true), dtype=tf.float32)
		return tf.reduce_mean(delta, axis=[0,1])[0]

	def my_mae_h(self, y_true, y_pred):
		delta = tf.cast(tf.math.abs(y_pred - y_true), dtype=tf.float32)
		return tf.reduce_mean(delta, axis=[0,1])[1]


	def my_scheduler(self, epoch, lr):
		if epoch < 3:
			return lr
		else: 
			if self.version=='a':
				return lr*tf.math.exp(-0.21)
			else:
				return lr


	def evaluate(self, X_test):
		"""
		It returns loss, mae_t, mae_h
		"""
		print("Evaluting the model...")
		return self.model.evaluate(X_test)

	def optimize(self, quantization, pruning, learning_rate, epochs, pruning_schedule, final_sparsity, initial_sparsity=0, schedule='polynomial'):
		"""
		Wrapper function to handle optimization methods.

		Parameters:
		- quantization: boolean; it indicates whether or not to apply quantization
		- pruning: boolean; it indicates whether or not to apply quantization
		- learning_rate: float; learning rate to apply in the training with pruning
		- epochs: int; number of epoch of the trian with pruning
		- pruning_schedule: string, 'constant' or 'polynomial'
		- final_sparsity: float; final sparsity in case of polynomial pruning schedule
		- initial_sparsity: float; initial sparsity for both polynomial and constant pruning schedule
		"""
		if pruning:
			self.pruning = True
			prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude 
			if schedule=='constant':
				schedule = tfmot.sparsity.keras.ConstantSparsity(initial_sparsity, 1000)
			else:
				schedule = tfmot.sparsity.keras.PolynomialDecay( 
					                            initial_sparsity=initial_sparsity,                                                             
					                            final_sparsity=final_sparsity,                                                                
					                            begin_step=0,                                                             
					                            end_step=np.ceil(len(self.X_train)/self.batch_size).astype(np.int32)*epochs - 100)
			self.model = prune_low_magnitude(self.model, 
											pruning_schedule = schedule)

			self.model.compile(loss = tf.keras.losses.MeanSquaredError(),
                            	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            	metrics = My_MeanAbsoluteError())
			self.model.summary()

			callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
						tf.keras.callbacks.LearningRateScheduler(self.my_scheduler, verbose=0),
						tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True),
						keras.callbacks.EarlyStopping(monitor='loss', patience = 4, verbose = 0, mode='auto', restore_best_weights=True)]
			self.model.fit(self.X_train,
                           epochs=epochs,
                           callbacks=callbacks)

		if quantization:
			self.quantization = True

	def apply_pruning_to_dense(self, layer):
		"""
		Helper function uses `prune_low_magnitude` to make only the 
		Dense layers we want to train with pruning.
		"""
		prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
		end_step = np.ceil(len(self.X_train) / self.batch_size).astype(np.int32) * self.epochs
		schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0, 
														final_sparsity=0.8,
														begin_step=0, 
														end_step=end_step, 
														power=2, 
														frequency=100)
		if isinstance(layer, tf.keras.layers.Dense) and (layer.name != 'final_dense'):
			return prune_low_magnitude(layer, pruning_schedule=schedule)
		
		return layer



	def save(self, out_dir='./saved_models'):
		"""
		This function saves the keras model on an output directory.
		Before saving, it removes the additional wrappers eventually added with pruning or clustering.

		Parameters:
		- out_dir: name of the output directory. Default = './saved_models'
		"""
		# remove the additional wrappers
		if self.pruning:
			self.model = tfmot.sparsity.keras.strip_pruning(self.model)
		
		# save the model on file
		self.saved_model_dir = out_dir
		run_model = tf.function(lambda x: self.model(x))
		concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
		self.model.save(self.saved_model_dir, signatures=concrete_func)


	def create_tflite(self, out_dir):
		"""
		It is needed to save the model before calling this function!
		This function will create the tflite file of the saved model, compress it and save it in the output directory.
		If 'Quantization' is True, before saving the model weights will be quantized.

		Parameters:
		- out_dir: string, output directory. Default='./TFLite/model_MLP.tflite'
		"""
		converter = tf.lite.TFLiteConverter.from_saved_model(self.saved_model_dir)

		if self.quantization:
			converter.optimizations = [tf.lite.Optimize.DEFAULT]
			#converter.representative_dataset = self.representative_data_gen
		tflite_model = converter.convert()    

		# save the tflite model on file
		with open(out_dir, 'wb') as fp:
			tflite_model = zlib.compress(tflite_model)
			fp.write(tflite_model)



if __name__ == "__main__":

	seed = 31
	np.random.seed(seed)
	tf.random.set_seed(seed)

	parser = argparse.ArgumentParser()
	parser.add_argument('--version', type=str, required=True, help='exercise version: a or b')
	args = parser.parse_args()

	LABEL_OPTIONS = 2
	INPUT_WIDTH = 6

	if args.version=='a':
		OUT_STEPS = 3
		model_params = {'version': 'a',
				'out_steps':3,
				'batch_size': 256,
				'alpha':0.25}
		LR = 0.01
		epochs = 5
		opt_params = {'quantization': True,
					'pruning': True,
					'pruning_schedule': 'polynomial',
					'initial_sparsity': 0,
					'final_sparsity': 0.65,
					'learning_rate': 1e-3,
					'epochs': 20}

	elif args.version=='b':
		OUT_STEPS = 9
		model_params = {'version': 'b',
				'out_steps': 9,
				'batch_size': 256,
				'alpha': 0.4}
		LR = 1e-3
		epochs = 5
		opt_params = {'quantization': True,
					'pruning': True,
					'pruning_schedule': 'polynomial',
					'initial_sparsity': 0,
					'final_sparsity': 0.65,
					'learning_rate': 5e-4,
					'epochs': 15}
	else:
		raise Exception("Invalid version argument. Select version 'a' or 'b'")


	# read the data -------------------------
	zip_file = tf.keras.utils.get_file(origin = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",\
										fname = "jena_climate_2009_2016.csv.zip", extract = True, cache_dir=".",cache_subdir="data")
	csv_path, _ = os.path.splitext(zip_file)
	df = pd.read_csv(csv_path)

	# DATA PREPROCESSING --------------------

	# select the columns of interest
	column_indices = [2,5]    # 2 = temperature, 5 = humidity
	columns = df.columns[column_indices]
	data = df[columns].values.astype(np.float32)

	n = len(data)
	train_data = data[0:int(n*0.7)]
	val_data = data[int(n*0.7):int(n*0.9)]
	test_data = data[int(n*0.9):]

	mean = train_data.mean(axis=0)	
	std = train_data.mean(axis=0)

	generator = WindowGenerator(INPUT_WIDTH, mean, std, LABEL_OPTIONS, OUT_STEPS)
	train_ds = generator.make_dataset(train_data, True)
	val_ds = generator.make_dataset(val_data, False)
	test_ds = generator.make_dataset(test_data, False)
	print("-- Done preprocessing")


	# MODEL CONTSTRUCTION ---------------------------------------       
	model = MyModel(input_width=INPUT_WIDTH, 
					label_options=LABEL_OPTIONS, 
					X_train=train_ds,
					**model_params)
	
	history = model.train_base_model(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
						loss_function=tf.keras.losses.MeanSquaredError(),
						epochs=epochs)


	# PRUNING ---------------------------------------------------
	model.optimize(**opt_params)
	loss, mae = model.evaluate(test_ds)
	model.save(out_dir='./pruned_model')

	# CREATE TFLITE --------------------------------------------- 
	tflite_model_dir = f"./TFLite/Group7_th_{args.version}.tflite.zlib"
	model.create_tflite(out_dir=tflite_model_dir)


	print("# output steps:", OUT_STEPS)			
	print("T MAE:", mae[0])
	print("Rh MAE:", mae[1])
	print("TFLite Size: {} kB".format(os.path.getsize(tflite_model_dir)/1000))

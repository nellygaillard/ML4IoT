import argparse
import pandas as pd
from datetime import datetime
import tensorflow as tf
import os

def normalize_func(df):
	"""Normalization of humidity and temperature values in the dataframe
	
	Parameters
	----------
	- df: pandas.DataFrame (dataframe having as columns date_time, temperature and humidity)
	
	Output
	------
	- pandas.DataFrame (dataframe with the same structure of the input dataframe, having the temperature and humidity columns been normalized)"""

	for col in ['temperature', 'humidity']:
		max = df[col].max()
		min = df[col].min()
		df[col] = (df[col] - min)/(max - min)
	return df

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def date_parser(date,time):
	"""From a date (format: year-month-day) and a time value (format: hour:minute:second) it returns a posix timestamp as integer.
	
	Parameters
	----------
	- date: object
	- time: object
	
	Output
	------
	- int (posix timestamp)"""
	return int(datetime.strptime(str(date + " " + time), "%d/%m/%Y %H:%M:%S").timestamp())

def write_file(normalize_flag, filename, df):
	"""Build a TFRecord Dataset
	
	Parameters
	----------
	- normalize_flag: bool (whether to normalize the data related to temperature and humidity or not)
	- filename: string (name of the output file without the extension)
	- df: pandas.Dataframe (dataframe with the data)"""

	with tf.io.TFRecordWriter("{}.tfrecord".format(filename)) as writer: #e.g. test.tfrecord
		for i in range(len(df)):
			datetime_vals = int64_feature(df.date_time[i])
			if normalize_flag:
				temp_vals = float_feature(df.temperature[i])
				humidity_vals = float_feature(df.humidity[i])
			else:
				temp_vals = int64_feature(df.temperature[i])
				humidity_vals = int64_feature(df.humidity[i])
			mapping = {'date_time': datetime_vals, 'temperature': temp_vals, 'humidity': humidity_vals}
			example = tf.train.Example(features=tf.train.Features(feature=mapping))
			writer.write(example.SerializeToString())

	
if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('--input',type=str, required=True, help='name of the input csv file')
	parser.add_argument('--output',type=str, required=True, help='name of the output csv file')
	parser.add_argument('--normalize',action='store_true', help='normalize flag')
	args = parser.parse_args()
	
	df = pd.read_csv(args.input, sep=',', header=None, names = ['date','time','temperature','humidity'], parse_dates=[['date','time']],date_parser=date_parser)

	if (args.normalize):
		df = normalize_func(df)

	write_file(args.normalize, args.output, df)

	file_size = os.path.getsize("{}.tfrecord".format(args.output))
	print(f"{file_size}B")
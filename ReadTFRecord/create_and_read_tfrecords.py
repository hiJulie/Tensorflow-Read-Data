"""
convert() 负责把数据转化成 TFRecords 格式;
read_and_decode() 读取一个样本;
input_pipeline() 负责将输入样本顺序打乱 (shuffle=True)，并生成 mini_batch的训练样本。
read_test() 是一个从文件中批读出的例子程序
"""

import numpy as np
import tensorflow as tf
from six.moves import xrange
import cv2

def convert():
	scalars = np.array([1, 2, 3])
	
	vectors = np.array([[0.1, 0.1, 0.1],
	                    [0.2, 0.2, 0.2],
	                    [0.3, 0.3, 0.3]])
	
	matrices = np.array([np.array((vectors[0], vectors[0])),
	                     np.array((vectors[1], vectors[1])),
	                     np.array((vectors[2], vectors[2]))])
	
	# shape of image：(720,1280,3)
	img = cv2.imread('img_1.jpg')
	tensors = np.array([img, img, img])
	print(img.shape)

	
	# open TFRecord file
	writer = tf.python_io.TFRecordWriter('%s.tfrecord' % 'test')

	# we are going to write 3 examples，each example has 4 features：scalar, vector, matrix, tensor
	for i in range(3):
		# create dictionary
		features = {}
		# write scalar ，type Int64，"value=[scalars[i]]" makes it to list
		features['scalar'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[scalars[i]]))

		# write vector，type float，it is list，so "value=vectors[i]"
		features['vector'] = tf.train.Feature(float_list=tf.train.FloatList(value=vectors[i]))

		# write matrix，type float，but its rank =2，tf.train.FloatList only takes list, so we can flatten it to list
		features['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value=matrices[i].reshape(-1)))
		# however the shape info will disappear. we can save shape as vector here
		features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=matrices[i].shape))

		# write tensor，type float，rank =3，another way is to convert it to string
		features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
		# save shape (608,816,3)
		features['tensor_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensors[i].shape))

		# feed dictionary to tf.train.Features
		tf_features = tf.train.Features(feature=features)
		# get an example
		tf_example = tf.train.Example(features=tf_features)
		# serialize the example
		tf_serialized = tf_example.SerializeToString()
		# write
		writer.write(tf_serialized)
	# close
	writer.close()


def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()

	_, serialized_example = reader.read(filename_queue)

	# 创建样本解析字典：该字典存放着所有feature的解析方式，key为feature名，value为feature的解析方式。
	dics = {'scalar': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

	        'vector': tf.FixedLenFeature(shape=(3,), dtype=tf.float32),

	        # 使用 VarLenFeature来解析
	        'matrix': tf.VarLenFeature(dtype=tf.float32),

	        # tensor在写入时 使用了toString()，shape是()
	        # 但这里的type不是tensor的原type，而是字符化后所用的tf.string，随后再回转成原tf.uint8类型
	        'tensor': tf.FixedLenFeature(shape=(), dtype=tf.string)}

	parsed_example = tf.parse_single_example(serialized_example, dics)

	# 解码字符
	parsed_example['tensor'] = tf.decode_raw(parsed_example['tensor'], tf.uint8)
	# 稀疏表示 转为 密集表示
	parsed_example['matrix'] = tf.sparse_tensor_to_dense(parsed_example['matrix'])

	# 转变matrix形状
	parsed_example['matrix'] = tf.reshape(parsed_example['matrix'], [2, 3])

	# 转变tensor形状
	parsed_example['tensor'] = tf.reshape(parsed_example['tensor'], [720, 1280, 3])

	return parsed_example

def input_pipeline(filename_queue, batch_size, num_epochs=None):
	parsed_example = read_and_decode(filename_queue)

	# min_after_dequeue defines how big a buffer we will randomly sample
	#   from -- bigger means better shuffling but slower start up and more
	#   memory used.
	# capacity must be larger than min_after_dequeue and the amount larger
	#   determines the maximum we will prefetch.  Recommendation:
	# capacity = min_after_dequeue + 3 * batch_size,

	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size
	parsed_example_batch = tf.train.shuffle_batch(parsed_example, batch_size=batch_size, capacity=capacity,
	                                              min_after_dequeue=min_after_dequeue)
	return parsed_example_batch


def read_test():
	filenames = ['test.tfrecord']
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=2, shuffle=True)
	parsed_example_batch = input_pipeline(filename_queue, batch_size=3, num_epochs=3)
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	with tf.Session() as sess:

		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		try:
			while not coord.should_stop():  # Check if stop was requested.True if a stop was requested.
				# reader每次读取一个文件中的一行数据，在session中运行这个tensor才能取到实际的值
				for i in xrange(7):
					example = sess.run(parsed_example_batch['matrix'])
					print(example)

		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			# 请求线程结束
			coord.request_stop()

		# 等待线程终止
		coord.join(threads)


if __name__ == '__main__':
	convert()
	print("convert finished")
	read_test()
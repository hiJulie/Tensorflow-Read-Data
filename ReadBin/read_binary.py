import tensorflow as tf
import os
from six.moves import xrange

IMAGE_SIZE = 32
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000#训练数据共有50000个
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000#测试数据共有10000个

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 12800,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './ReadBin/',
                           """Path to the CIFAR-10 data directory.""")#给相对路径

def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """
  #定义一个类
  class CIFAR10Record(object):
    pass
  
  result = CIFAR10Record()#类的实例

  #result.height表示实例具有属性height
  result.height = 32
  result.width = 32
  result.depth = 3

  label_bytes = 1  # 2 for CIFAR-100
  image_bytes = result.height * result.width * result.depth
  
  #每条纪录包含label跟image，长度固定为record_bytes
  record_bytes = label_bytes + image_bytes

  # 选择针对输入文件格式的阅读器、解码器
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)#读取出来的数据类型为string

  #将读取出来的string数据类型的文件解码，并解码成uint8数据类型
  record_bytes = tf.decode_raw(value, tf.uint8)
  
  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  result.uint8image = tf.cast(result.uint8image, tf.float32)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', image_batch)
  label_batch = tf.reshape(label_batch, [batch_size])

  return image_batch, label_batch

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  
  #生成文件名列表
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # 生成文件名队列
  filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)#总共参与训练的数据总数为：每个epoch的数据量*num_epochs=50000*1

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Set the shapes of tensors.
  read_input.uint8image.set_shape([height, width, 3])
  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)
  image_batch, label_batch = _generate_image_and_label_batch(read_input.uint8image, read_input.label,
                                  min_queue_examples, batch_size,
                                  shuffle=False)

  # Generate a batch of images and labels by building up a queue of examples.
  return image_batch, label_batch#每次从样本队列中取出batch=128的数据


if __name__ == '__main__':
	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	image_batch, label_batch = inputs(eval_data=False,data_dir=data_dir,batch_size=FLAGS.batch_size)#最后送入网络进行计算的图片的数据为float，标签为int
	
	# num_epochs是个局部变量，需要进行初始化
	init = tf.local_variables_initializer()
	
	with tf.Session() as sess:
		sess.run(init)
		
		# 线程协调器,用来管理之后在Session中启动的所有线程;
		coord = tf.train.Coordinator()
		
		# 开始在图表中收集队列运行器
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		"""
			在我们使用tf.train.string_input_producer创建文件名队列后，整个系统其实还是处于“停滞状态”的。
			此时如果我们开始计算，因为队列中什么也没有，计算单元就会一直等待，导致整个系统被阻塞。
			因此需要调用tf.train.start_queue_runners()，这个函数将会启动所有线程，开始填充样本到队列中。
			函数参数：sess使用默认会话，coord用于协调已经启动的线程；函数返回值：所有线程的列表
			"""
		try:
			while not coord.should_stop():# Check if stop was requested.True if a stop was requested.
				# reader每次读取一个文件中的一行数据，在session中运行这个tensor才能取到实际的值
				for i in range(10):
					# Retrieve a single instance:
					features_batch_, labels_batch_ = sess.run(
						[image_batch, label_batch])  # 每次取出batch=128的数据，取10次，则一共从样本队列中取出128*10个样本
					print(features_batch_.shape, labels_batch_)
		
		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			# 请求线程结束
			coord.request_stop()
			
		# 等待线程终止
		coord.join(threads)
		


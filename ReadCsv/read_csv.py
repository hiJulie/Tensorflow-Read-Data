import tensorflow as tf

# 生成文件名列表
filename = ["Iris-test.csv", "Iris-train.csv"]

# 生成文件名队列
filename_queue = tf.train.string_input_producer(filename, num_epochs=5, shuffle=True)

# 选择针对输入文件格式的阅读器、解码器
reader = tf.TextLineReader(skip_header_lines=1)  # 第一行跳过
key, value = reader.read(filename_queue)

record_defaults = [[0.], [0.], [0.], [0.], ['']]#根据训练数据的数据类型来设置
col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)#每次读取一条数据
features = tf.stack([col1, col2, col3, col4])  # 前四列为特征data，最后一列是标签label

#将标签Iris-setosa，Iris-versicolor，Iris-virginica分别用数值0,1,2代替；默认值为-1
labels = tf.case({
	tf.equal(col5, tf.constant('Iris-setosa')): lambda: tf.constant(0),
	tf.equal(col5, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
	tf.equal(col5, tf.constant('Iris-virginica')): lambda: tf.constant(2),
}, lambda: tf.constant(-1), exclusive=True)

"""
	tf.case(pred_fn_pairs, default, exclusive=False, name="case")相当于switch语句
	参数：
	pred_fn_pairs：大小为N的字典或pairs的列表。每对包含一个布尔标量tensor和一个python可调用函数项，当条件为True将返回对应的函数项创建的tensors。pred_fn_pairs的可调用函数返回值以及默认值（如果提供的话）都应该返回相同形状和类型的张量。

	如果exclusive==True，则计算pred_fn_pairs中的所有布尔值，如果有多个True，则引发异常。
	如果exclusive==False，则执行在求值为True的第一个布尔值处停止，并且立即返回由相应函数生成的张量（tensors）。如果没有一个布尔值评估为true，则此操作返回默认生成的张量。
	"""

batch_size = 100
min_after_dequeue = 10#在一组batch元素出队后，队列里面需要剩余元素的最小数
capacity = min_after_dequeue + 3 * batch_size#这个值一定要比min_after_dequeue大，表示队列容量即队列中元素的最大数量
# 将队列中数据打乱后再读取出来
features_batch, labels_batch = tf.train.shuffle_batch([features, labels], batch_size=batch_size,
                                                      capacity=capacity, min_after_dequeue=min_after_dequeue)

"""
tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)：[example, label]表示样本和样本标签，这个可以是一个样本和一个样本标签，batch_size是返回的一个batch样本集的样本个数。capacity是队列中的容量。这主要是按顺序组合成一个batch。
tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue)。这里面的参数和上面的一样的意思。不一样的是这个参数min_after_dequeue，一定要保证这参数小于capacity参数的值，否则会出错。这个代表队列中的元素大于它的时候就输出乱的顺序的batch。也就是说这个函数的输出结果是一个乱序的样本排列的batch，不是按照顺序排列的。
上面的函数返回值都是一个batch的样本和样本标签，只是一个是按照顺序，另外一个是随机的
"""

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
		while not coord.should_stop():  # Check if stop was requested.True if a stop was requested.
			# reader每次读取一个文件中的一行数据，在session中运行这个tensor才能取到实际的值
			for i in range(10):
				# Retrieve a single instance:
				features_batch_, labels_batch_ = sess.run([features_batch, labels_batch])
				print(features_batch_.shape, labels_batch_)
				
	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# 请求线程结束
		coord.request_stop()

	# 等待线程终止
	coord.join(threads)
	
	

	



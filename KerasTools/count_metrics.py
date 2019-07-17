import tensorflow as tf
from keras import backend as K
from keras.applications.mobilenet import MobileNet

run_metadata = tf.RunMetadata()

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)

    model = MobileNet(alpha=1.0, weights=None, input_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)))

    opt = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op_name', options=opt)

    opt = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    param_count = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op_name', options=opt)

    print('flops:', flops.total_float_ops)
    print('param count:', param_count.total_parameters)

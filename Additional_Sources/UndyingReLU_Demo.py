import numpy as np
import tensorflow as tf
import UndyingReLU

def dense(input_layer, units, name=None):
    return tf.layers.dense(
        inputs=input_layer,
        units=units,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        name=name,
        activation=None)

def get_tensors_by_prefix(prefix):
    tnsr_names = [n.name + ':0' for n in tf.get_default_graph().as_graph_def().node if
                           n.name.startswith(prefix)]
    tensors = [tf.get_default_graph().get_tensor_by_name(nm) for nm in tnsr_names]
    return tensors

DATA_DIMENSION = 4
NUM_SAMPLES = 11

# optional: reset UndyingReLU parameters:
UndyingReLU.parameters['decay_rate'] = 0.999
UndyingReLU.parameters['required_active_percent'] = 10

sess = tf.InteractiveSession()

# define the network
# ------------------
data = tf.placeholder(tf.float32,[None,DATA_DIMENSION], name='data')
label = tf.placeholder(tf.float32,[None,DATA_DIMENSION], name='label')
dense_lyr1 = dense(data, DATA_DIMENSION, name="dense1")

# An Undying ReLU layer:
relu1 = UndyingReLU.relu(dense_lyr1, name='relu1')

dense_lyr2 = dense(data, DATA_DIMENSION, name="dense2")

# Another Undying ReLU layer:
relu2 = UndyingReLU.relu(dense_lyr2, name='relu2')

prediction = dense(relu2, DATA_DIMENSION, name="prediction")
loss = tf.losses.mean_squared_error(label, prediction)
solver = tf.train.AdamOptimizer(0.1).minimize(loss)

sess.run(tf.global_variables_initializer())

for i in xrange(500):

    cur_input = np.reshape(np.arange(DATA_DIMENSION * NUM_SAMPLES), [NUM_SAMPLES,DATA_DIMENSION])
    cur_label = np.flip(cur_input, axis=1)
    feed_dict = { data: cur_input, label: cur_label }

    if np.mod(i,200) == 1:
        UndyingReLU.resurrect(sess, feed_dict)
        print("< adjusting life support >")

    feed_dict = UndyingReLU.add_to_feed_dict(feed_dict)

    if np.mod(i,200) in [0, 1,199]:
        casualty_report, _, _, _, _ = UndyingReLU.measure_dead_neurons(sess, feed_dict, after_life_support=False)
        casualty_report_with_life_support, _, _, _, _ = UndyingReLU.measure_dead_neurons(sess, feed_dict, after_life_support=True)
        print "iteration %d" % i
        print "Num of Dead Neurons without life support: %d" % casualty_report['total_dead']
        print "Num of Dead Neurons with life support: %d" % casualty_report_with_life_support['total_dead']

    sess.run([solver], feed_dict=feed_dict)




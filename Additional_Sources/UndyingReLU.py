import tensorflow as tf
import numpy as np

__doc__ = """
===================
Undying ReLU layer Module
===================
Created October 2017 by Amnon Drory,
as part of PhD research at School
of Electrical Engineering,
Tel-Aviv University.

--------
Overview
--------
ReLU units suffer from a phenomenon known as "dead neurons". 
A dead neuron is a neuron whose output values are always negative, and
which is followed by a ReLU. On the forward pass, thois relu sets
all these values to zero. In back propagation, the ReLU sets all
the gradients flowing into the dead neuron to zero. As a result, 
The weights of the dead neuron are not changed, at it remains dead 
indefinitely.
The Undying ReLU layer Module seeks to solve this problem by adding an
"life support" layer before the ReLU, that ensures that atleast 
some of the values entering the ReLU will be possible. To clarify:
for 100% of samples: [dead neuron output] < 0
but:
for p% of samples: [dead neuron output] + [life support] > 0
(where p is a parameter, e.g. 5).

By using UndyingReLU layers instead of ReLU layers, the dead 
neuron problem is prevented. 

-------
Caveats:
-------
* The definition of dead neurons specifies that for _all samples_, 
the output of the neuron is negative. However, to test all possible 
samples may be prohibitively time consuming. Instead, we assume that 
a representative mini-batch will be used to test which neurons are
dead.
* This module is implemented to work with tensorflow's feed_dict 
method of feeding data. It is not compatible with other methods.

-----
Usage:
-----
See UndyingReLU_Demo.py for a detailed example.
Generally, using UndyingReLU requires 4 changes to your code:
1. add the line:
    import UndyingReLU
2. Replacing all definition of relu layers with UndyingReLU layers. 
   e.g. replace:
    relu1 = tf.nn.relu(input_layer)
   with:
    relu1 = UndyingReLU.relu(input_layer)
3. At every training iteration, after creating a feed_dict, call:
    feed_dict = UndyingReLU.add_to_feed_dict(feed_dict)
4. Every k iterations (e.g. k=1000) call:
    UndyingReLU.resurrect(session, feed_dict)
   which measures which neurons are dead and adjusts the "life 
   support" layer accordingly. 

"""


parameters = { 'decay_rate': 1.0, # (0 to 1) Set to a value <1 to
                # have the life support values gradually shrink over time.
                # 1.0 means no decay, 0.0 means immediate decay
               'required_active_percent': 5} # (0 to 100) The life support
                # addition is set so that this percent of samples will
                # have positive outputs

life_support_data = {}

def measure_dead_neurons(session, feed_dict, after_life_support=False):
    """
    Measure the number of dead neurons in each Undying ReLU layer
    
    :param session: a TensorFlow session 
    :param feed_dict: defines the group of samples used to define which neurons are dead. 
                      (see documentation of session.run() for definition of feed_dict)
    :param after_life_support: If True, then we measure the output of a neuron AFTER adding
                               the life support offset to it. Even so, a neuron may be dead, for 
                               example for the following reasons:  
                               1. Some time has passed since the support vector was last set, OR
                               2. The life support vector was set using a different feed_dict than
                                  the one used in this function.

    :return:     
    casualty_report - see resurrect()
    num_dead - for each Undying ReLU layer, the number of dead neurons.
    neuron_is_dead - for each Undying ReLU layer, for each neuron (=channel), whether it is dead  
    life_support_tensors - the life support tensor corresponding to each Undying ReLU layer 
    arrays - for each Undying ReLU layer, an ndarray containing the input for this 
                   layer (or the resurrected version of those if after_life_support==True) 
    """

    feed_dict = add_to_feed_dict(feed_dict, allow_side_effects=False)

    life_support_tensors = tf.get_collection('undying_relu_life_support_tnsrs')

    if after_life_support:
        tensors = [life_support_data[t]['resurrected_tnsr'] for t in life_support_tensors]
    else:
        tensors = [life_support_data[t]['input_tnsr'] for t in life_support_tensors]

    arrays = session.run(tensors, feed_dict=feed_dict)

    casualty_report = {}
    neuron_is_dead = []
    num_dead = []
    total_dead = 0
    total_neurons = 0
    for array, life_support_tnsr in zip(arrays, life_support_tensors):
        is_active = (array > 0).astype(float)
        num_active = np.sum(is_active, axis=tuple(range(np.ndim(is_active)-1)), keepdims=True)
        neuron_is_dead.append(num_active == 0)
        num_dead.append(neuron_is_dead[-1].sum())
        cur_total_neurons = np.prod(array.shape[1:])
        name = life_support_tnsr.name.replace('/life_support_tnsr:0','')
        casualty_report[name] = (num_dead[-1], # num dead neurons
                                 cur_total_neurons) # total neurons in layer
        total_dead += num_dead[-1]
        total_neurons += cur_total_neurons

    casualty_report['total_dead'] = total_dead
    casualty_report['total_neurons'] = total_neurons
    casualty_report['num_samples'] = arrays[0].shape[0]

    details = {}

    details['num_dead'] = num_dead
    details['neuron_is_dead'] = neuron_is_dead
    details['life_support_tensors'] = life_support_tensors
    details['arrays'] = arrays
    details['casualty_report'] = casualty_report
    return casualty_report, details

def resurrect(session=None, feed_dict=None, measured_dead_neuron_details=None):
    """
    Measure which neurons are dead, and set the life support vector to bring them back to life.
    
    :param session: a TensorFlow session 
    :param feed_dict: defines the group of samples used to define which neurons are dead. 
                      (see documentation of session.run() for definition of feed_dict)
    :param measured_dead_neuron_details: the second output of measure_dead_neurons().
                    If not supplied, then both session and feed_dict must be supplied.
    :return: casualty_report, a dictionary detailing the number of dead neurons for each UndyingRelu.
             example and explanation:
                {u'relu2': (3, 4), u'relu1': (3, 4), 'total_dead': 6, 'total_neurons': 8, 'num_samples': 11}
              u'relu2': (3, 4) - in the UndyingReLU layer named 'relu2', there are 3 dead neurons, out of a total of 4 neurons in this layer
              'total_dead': 6 - there are overall 6 dead neurons in the entire network
              'num_samples': 11 - the mini-batch used to define which neurons are dead contained 11 samples.
    
    example usage
    -------------
        if np.mod(iter,1000) == 0: # to save calculations, only measure dead neurons every 1000 iterations
            casualty_report = UndyingReLU.resurrect(sess, feed_dict)
            if casualty_report['total_dead'] > 3:
                learning_rate *= 0.5
            print(casualty_report)        
    """
    if measured_dead_neuron_details is None:
        _, details = measure_dead_neurons(session, feed_dict)
    else:
        details = measured_dead_neuron_details 

    num_dead = details['num_dead']
    neuron_is_dead = details['neuron_is_dead']
    life_support_tensors = details['life_support_tensors']
    arrays = details['arrays']
    casualty_report = details['casualty_report']

    for i in xrange(len(life_support_tensors)):

        life_support_tnsr = life_support_tensors[i]
        vector = life_support_data[life_support_tnsr]['vector']
        vector *= 0
        life_support_data[life_support_tnsr]['time_since_resurrection'] = 0

        if num_dead[i] > 0:
            array = arrays[i]
            dead_inds = np.where(neuron_is_dead[i])
            for inds_in_vector in zip(*dead_inds):
                inds_in_array = (slice(None),) + inds_in_vector[1:]
                samples = array[inds_in_array]
                assert (samples<0).all(), "found non negative values in supposedly dead neuron"
                addition = np.percentile(np.abs(samples), parameters['required_active_percent'])
                vector[inds_in_vector] = addition

    return casualty_report

def add_to_feed_dict(feed_dict, allow_side_effects=True):
    """
    Add feeds for life support tensors into feed_dict
        
    :param feed_dict: see documentation of session.run() for definition. Should
                      already contain all data (e.g. inputs, labels), except for
                      what is added by this function.
    :param: allow_side_effects: if True (default), may also decay 
                                the value of the life support tensor 
    :return: feed_dict, ready to be used with session.run()
    
    example usage
    -------------
        feed_dict = { data: cur_input, label: cur_label }
        feed_dict = UndyingReLU.add_to_feed_dict(feed_dict)
    """

    life_support_tensors = tf.get_collection('undying_relu_life_support_tnsrs')
    if set(life_support_tensors).issubset(set(feed_dict.keys())):
        return feed_dict # this function has been called for the second time. do nothing

    for life_support_tnsr in life_support_tensors:
        life_support_vector = life_support_data[life_support_tnsr]['vector']
        time_since_resurrection = life_support_data[life_support_tnsr]['time_since_resurrection']
        if allow_side_effects:
            if (time_since_resurrection > 0):
                life_support_vector *= parameters['decay_rate']
            time_since_resurrection += 1
        feed_dict[life_support_tnsr] = life_support_vector

    return feed_dict

def relu(input_tnsr, name=None):
    """
    Define an Undying ReLU layer
        
    :param input_tnsr: the input layer 
    :param name: optional, name for the layer
    :return: The newly created Undying ReLU layer
    
    example usage
    -------------
        relu1 = UndyingReLU.relu(dense1)
    """

    # Define life support tensor
    vector_shape = input_tnsr.shape.as_list()
    vector_shape[0] = 1
    default_life_support_tnsr = tf.zeros(shape=vector_shape,
                                         dtype=input_tnsr.dtype,
                                         name=name + '/default_life_support_tnsr')
    life_support_tnsr = tf.placeholder_with_default(
        input=default_life_support_tnsr,
        shape=vector_shape,
        name=name + '/life_support_tnsr')

    tf.add_to_collection('undying_relu_life_support_tnsrs', life_support_tnsr)

    # Define life support vector (will be fed to life-support-tensor using feed_dict)
    vector_shape = np.array(input_tnsr.shape.as_list())
    vector_shape[:-1] = 1
    vector_shape = list(vector_shape)
    life_support_vector = np.zeros(shape=vector_shape,
                                   dtype=input_tnsr.dtype.as_numpy_dtype)

    # Define layer after resurrection (=addition of constant to prevent relu from dying)
    resurrected = tf.add(input_tnsr, life_support_tnsr, name=name + '/resurrected')

    # Define relu layer
    relu = tf.nn.relu(resurrected, name=name + '/relu')

    # Record all necessary parts of the Undying ReLU layer
    life_support_data[life_support_tnsr] = {}
    life_support_data[life_support_tnsr]['vector'] = life_support_vector
    life_support_data[life_support_tnsr]['input_tnsr'] = input_tnsr
    life_support_data[life_support_tnsr]['resurrected_tnsr'] = resurrected
    life_support_data[life_support_tnsr]['time_since_resurrection'] = np.inf

    return relu

def reset():
    """
    Reset the life support vectors to zero
    
    :return: 
    """
    for life_support_tnsr in life_support_data.keys():
        life_support_data[life_support_tnsr]['time_since_resurrection'] = np.inf
        life_support_data[life_support_tnsr]['vector'] *= 0


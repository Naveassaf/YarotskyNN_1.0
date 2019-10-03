import tensorflow as tf
import numpy as np
import Additional_Sources.UndyingReLU as UDrelu

# CONSTANTS
from x_squared import USE_UNDYING_RELU
BUMP_FUNCTION_WIDTH = 4

class Bump():
    def __init__(self, N, bump_index, naming_postfix, input_placeholder=None, graph=None, yarotsky_initialization=True,
                 ud_relus=USE_UNDYING_RELU, trainable=True):
            # Naming convention prefix to prevent naming clashes with other blocks in tf_graph
            self.naming_postfix = naming_postfix

            # Dictionary with key[i] holding a dictionary of all weights, biases and activations of net rep. bump_function_i
            self.graph_dict = {}

            # Prepares an empty dictionary to be used to store all place holders in the net.
            self.feed_dict = {}

            # Decide if to use Undying or normal, tf relus
            self.use_ud_relus = ud_relus

            # Initialize parameters used to calculate the "shift" and horizontal scaling of the bump function
            self.bump_center = float(bump_index) / float(N)
            self.relu_offset = 1.0 / (3 * float(N))

            self.index = bump_index

            # For TF interface. Sets graph passed to constructor as the x^2 object's TF graph or creates graph if not passed
            if graph == None:
                self.tf_graph = tf.Graph()
            else:
                self.tf_graph = graph

            # Varialble used to indicate whether variables are initialized to match Yarotsky's formulas or randomly
            self.yarotsky_initialization = yarotsky_initialization

            # Create empty dict to save variables for pickling trained models. Dict (key, val) = (name, tf.variable)
            self.variable_dict = {}

            # If False, the bumps wioll retain their original shape throughout entire training session
            self.trainable = trainable

            # Prepare input placeholder
            with self.tf_graph.as_default():
                # Create place holder to be used to feed network its inputs
                if input_placeholder == None:
                    self.input_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
                else:
                    self.input_placeholder = input_placeholder

            self.build_bump_graph()

    def build_bump_graph(self):

        with self.tf_graph.as_default():
            self.first_weights = self.get_bump_weights(first=True, name=self.naming_postfix+'_weights_1')

            first_matmul = tf.matmul(self.input_placeholder, self.first_weights, name=self.naming_postfix+ '_matmul_1')

            self.first_biases = self.get_bump_biases(name=self.naming_postfix+'_biases')

            adder = tf.add(first_matmul, self.first_biases, name=self.naming_postfix+'_adder' )

            # If learning net, append batch normalization layer before ReLus
            if self.use_ud_relus:
                relus = UDrelu.relu(input_tnsr=adder, name=self.naming_postfix + '_relu')
            else:
                relus = tf.nn.relu(adder, name=self.naming_postfix + '_relu')

            self.second_weights = self.get_bump_weights(first=False, name=self.naming_postfix+'_weights_2')

            self.final_output = tf.matmul(relus, self.second_weights, name=self.naming_postfix+ '_matmul_2')

    def get_bump_weights(self, first, name):
        '''
        Function used to create weights object for bump function nets.

        :param first: Boolean which indicates if this is the first or second layer's weights within this bump function
        :param name: The name of the current weights TF object being created.
        :return: TF variable object.
        '''

        # Prep variable initializer function
        if self.yarotsky_initialization:
            initer = self.get_bump_weights_yarotsky_init(first, name)
        else:
            initer = tf.contrib.layers.xavier_initializer()

        if first:
            # Create variable for first weights in current neuron
            if self.yarotsky_initialization:
                var = tf.get_variable(name=name, dtype=tf.float64, initializer=initer, trainable=self.trainable)
            else:
                var = tf.get_variable(name=name, dtype=tf.float64, shape=[1, BUMP_FUNCTION_WIDTH], initializer=initer, trainable=self.trainable)
        else:
            # Create variable for second weights in current neuron
            if self.yarotsky_initialization:
                var = tf.get_variable(name=name, dtype=tf.float64, initializer=initer, trainable=self.trainable)
            else:
                var = tf.get_variable(name=name, dtype=tf.float64, shape=[BUMP_FUNCTION_WIDTH, 1], initializer=initer, trainable=self.trainable)

        # Store in dictionary for saving trained net
        self.variable_dict[name] = var

        return var

    def get_bump_weights_yarotsky_init(self, first, name):
        '''
        Function used to create weights object for bump function nets when using Yarotsky's formula for init values.

        :param first: Boolean which indicates if this is the first or second layer's weights within this bump function NN
        :param name: The name of the current weights TF object being made.
        :return: TF constant object with values based on Yarotsky's formula.
        '''
        slope = 1.0/self.relu_offset

        if first:
            return tf.constant(value=np.array([1, 1, 1, 1]), shape=[1, BUMP_FUNCTION_WIDTH], dtype=tf.float64, name=name)
        else:
            return tf.constant(value=np.array([slope, -slope, -slope, slope]), shape=[BUMP_FUNCTION_WIDTH, 1], dtype=tf.float64, name=name)

    def get_bump_biases(self, name):
        '''
        Function used to create biases object for bump function net.

        :param first: Boolean which indicates if this is the first or second layer's biases within this bump function NN.
        :param name: The name of the current biases TF object being made.
        :return: var: TF variable object.
        '''

        # Prep variable initializer function (init to 1 or yarotsky biases)
        if self.yarotsky_initialization:
            initial = self.get_bump_biases_yarotsky_init(name)
        else:
            initial = tf.constant(1, shape=[1, BUMP_FUNCTION_WIDTH], dtype=tf.float64)

        # Create variable
        var = tf.get_variable(name=name, dtype=tf.float64, initializer=initial, trainable=self.trainable)

        # Store in dictionary for pickling
        self.variable_dict[name] = var

        return var

    def get_bump_biases_yarotsky_init(self, name):
        '''
        Function used to create biases object for bump function net when using Yarotsky's formula for initialization.

        :param name: The name of the current biases TF object being made.
        :return: TF constant object with values based on Yarotsky's formula.
        '''

        return tf.constant(value=np.array([-(self.bump_center-2*self.relu_offset), -(self.bump_center-self.relu_offset),
                                           -(self.bump_center+self.relu_offset), -(self.bump_center+2*self.relu_offset)]),
                                            shape=[1, BUMP_FUNCTION_WIDTH], dtype=tf.float64, name=name)
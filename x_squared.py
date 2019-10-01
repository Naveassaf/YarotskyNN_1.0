import tensorflow as tf
import numpy as np
import Additional_Sources.UndyingReLU as UDrelu


#CONSTANTS
COMPOSITE_NET_WIDTH = 3
USE_UNDYING_RELU = False

class XSquared():
    def __init__(self, max_degree, naming_postfix, input_placeholder=None, graph=None, yarotsky_initialization=False,
                 ud_relus=USE_UNDYING_RELU, trainable=True):
        # max_degree is the largest hat function degree to be used. This parameter affects the accuracy of the net
        self.max_degree = max_degree

        # Naming convention prefix to prevent naming clashes with other blocks in tf_graph
        self.naming_postfix = naming_postfix

        # A list which stores all the output nodes of the different degree hat functions. This list is used after
        # the initialization of the hat function nets in order to sum over all the hat function nets' outputs.
        self.hat_outputs = []

        # Dictionary with key[i] holding a dictionary of all weights, biases and activations of net rep. hat_function_i
        self.graph_dict = {}

        # Prepares an empty dictionary to be used to store all place holders in the net.
        self.feed_dict = {}

        # Decide if to use Undying or normal, tf relus
        self.use_ud_relus = ud_relus

        # Determines whether object's variables will be trainable (used for training bumps only in spline NN)
        self.trainable=trainable

        # For TF interface. Sets graph passed to constructor as the x^2 object's TF graph or creates graph if not passed
        if graph == None:
            self.tf_graph = tf.Graph()
        else:
            self.tf_graph = graph

        # Varialble used to indicate whether variables are initialized to match Yarotsky's formulas or randomly
        self.yarotsky_initialization = yarotsky_initialization

        # Create empty dict to save variables for pickling trained models. Dict (key, val) = (name, tf.variable)
        self.variable_dict = {}

        # Prepare input placeholder
        with self.tf_graph.as_default():
            # Create place holder to be used to feed network its inputs
            if input_placeholder == None:
                self.input_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
            else:
                self.input_placeholder = input_placeholder

            # Effectively, appends g_0(x), hat function of degree 0. Needed to implement Yarotsky's calculation
            self.hat_outputs.append(self.input_placeholder)

            # Places vector of ones in feed_dict. The value of the vector does not matter as much as the shape
            # which will be used to run the TF session when evaluating the net.
            self.feed_dict[self.hat_outputs[0]] = np.ones(shape=[1, 1], dtype='float64')

            # Set net's current output to be the input. Updated final_output is used to link hat function nets and eval net.
            self.final_output = self.input_placeholder
            self.graph_dict[0] = {'output': self.final_output}

            # Create neural network (append hat function net max_degree times)
            for current_degree in range(1, max_degree + 1):
                self.graph_dict[current_degree] = self.append_hat_function(current_degree)

            # Link all created hat functions (effectively one, high-degree function, but saving outputs along the way)
            self.graph_dict['linker'] = self.append_hat_linker()

    def append_hat_function(self, current_depth):
        '''
        This function is used to create the actual hat function to be appended to object's graph.

        :param current_depth: Indicates the location of the current hat function within the net. Used for naming.
        :return: graph_dict - graph of pointers to TF objects in the current hat function being created.
        '''

        # Postfix is a char of this hat function's degree. To be used for naming nodes created in this method
        postfix = str(current_depth)+self.naming_postfix

        # Local dictionary used to store pointers to TF objects created for the current hat function.
        graph_dict = {}

        # Append tensorflow nodes to current graph
        with self.tf_graph.as_default():
            # Store this hat function's inputs as net's current output
            graph_dict['input'] = self.final_output

            # Create TF objects, linked and initialized based on Yarotsky's formula
            graph_dict['weights_first'] = self.get_hat_weights(first=True, name='weights_first_' + postfix)

            graph_dict['biases_first'] = self.get_hat_biases(first=True, name='biases_first_' + postfix)

            graph_dict['matmul_first'] = tf.matmul(graph_dict['input'], graph_dict['weights_first'], name='matmult_first_' + postfix)

            graph_dict['adder'] = tf.add(graph_dict['matmul_first'], graph_dict['biases_first'], name='adder_' +  postfix)

            # If learning net, append batch normalization layer before ReLus
            if self.use_ud_relus:
                graph_dict['relu'] = UDrelu.relu(input_tnsr=graph_dict['adder'], name='relu_' + postfix)
            else:
                graph_dict['relu'] = tf.nn.relu(graph_dict['adder'], name='relu_' + postfix)

            graph_dict['weights_second'] = self.get_hat_weights(first=False, name='weights_second_' + postfix)

            graph_dict['matmul_second'] = tf.matmul(graph_dict['relu'], graph_dict['weights_second'],
                                                    name='matmult_second_' + postfix)

            graph_dict['output'] = graph_dict['matmul_second']

            # Append output of hat function to list of hat_function outputs. To be added to form final net's output.
            self.hat_outputs.append(graph_dict['output'])

            # Update current output to be output of current layer
            self.final_output = graph_dict['output']

        # Return dictionary holding pointers to all TF objects of all layers of hat function network created
        return graph_dict

    def get_hat_weights(self, first, name):
        '''
        Function used to create weights object for hat function nets.

        :param first: Boolean which indicates if this is the first or second layer's weights within this hat function NN
        :param name: The name of the current weights TF object being created.
        :return: TF variable object.
        '''

        # Prep variable initializer function
        if self.yarotsky_initialization:
            initer = self.get_hat_weights_yarotsky_init(first, name)
        else:
            initer = tf.contrib.layers.xavier_initializer()

        if first:
            # Create variable for first weights in current neuron
            if self.yarotsky_initialization:
                var = tf.get_variable(name=name, dtype=tf.float64, initializer=initer, trainable=self.trainable)
            else:
                var = tf.get_variable(name=name, dtype=tf.float64, shape=[1, COMPOSITE_NET_WIDTH], initializer=initer, trainable=self.trainable)
        else:
            # Create variable for second weights in current neuron
            if self.yarotsky_initialization:
                var = tf.get_variable(name=name, dtype=tf.float64, initializer=initer, trainable=self.trainable)
            else:
                var = tf.get_variable(name=name, dtype=tf.float64, shape=[COMPOSITE_NET_WIDTH, 1], initializer=initer, trainable=self.trainable)

        # Store in dictionary for saving trained net
        self.variable_dict[name] = var

        return var

    def get_hat_weights_yarotsky_init(self, first, name):
        '''
        Function used to create weights object for hat function nets when using Yarotsky's formula for init values.

        :param first: Boolean which indicates if this is the first or second layer's weights within this hat function NN
        :param name: The name of the current weights TF object being made.
        :return: TF constant object with values based on Yarotsky's formula.
        '''

        if first:
            return tf.constant(value=np.array([1, 1, 1]), shape=[1, COMPOSITE_NET_WIDTH], dtype=tf.float64, name=name)
        else:
            return tf.constant(value=np.array([2, -4, 2]), shape=[COMPOSITE_NET_WIDTH, 1], dtype=tf.float64, name=name)

    def get_hat_biases(self, first, name):
        '''
        Function used to create biases object for hat function net.

        :param first: Boolean which indicates if this is the first or second layer's biases within this hat function NN.
        :param name: The name of the current biases TF object being made.
        :return: var: TF variable object.
        '''

        # Prep variable initializer function (init to 1 or yarotsky biases)
        if self.yarotsky_initialization:
            initial = self.get_hat_biases_yarotsky_init(first, name)
        else:
            initial = tf.constant(1, shape=[1, COMPOSITE_NET_WIDTH], dtype=tf.float64)

        # Create variable
        var = tf.get_variable(name=name, dtype=tf.float64, initializer=initial, trainable=self.trainable)

        # Store in dictionary for pickling
        self.variable_dict[name] = var

        return var

    def get_hat_biases_yarotsky_init(self, first, name):
        '''
        Function used to create biases object for hat function net when using Yarotsky's formula for initialization.

        :param first: Boolean which indicates if this is the first or second layer's biases within this hat function NN.
        :param name: The name of the current biases TF object being made.
        :return: TF constant object with values based on Yarotsky's formula.
        '''

        if first:
            return tf.constant(value=np.array([0, -0.5, -1]), shape=[1, COMPOSITE_NET_WIDTH], dtype=tf.float64, name=name)
        else:
            return tf.constant(value=np.array([0]), shape=[1, 1], dtype=tf.float64, name=name)

    def append_hat_linker(self):
        '''
        Function called as the last step of the neural net's construction. Links outputs from all 'hat function' nets
        by using a matrix multiplication op.

        :return: graph_dict -
        '''

        # Init dictionary to store refrences to all new TF nodes
        graph_dict = {}

        # Open current graph and add weights, biases, and matmult node
        with self.tf_graph.as_default():
            graph_dict['inputs'] = tf.concat(self.hat_outputs, name='hat_outputs'+self.naming_postfix, axis=1)

            graph_dict['weights'] = self.get_linker_weights(name='weights_linker'+self.naming_postfix)

            graph_dict['matmul'] = tf.matmul(graph_dict['inputs'], graph_dict['weights'], name='matmult_linker'+self.naming_postfix)

            graph_dict['output'] = graph_dict['matmul']

        # Update current output to be output of linker layer
        self.final_output = graph_dict['output']

        return graph_dict

    def get_linker_weights(self, name):
        '''
        Function used to create weights object for the linker node.

        :param name: The name of the current weights TF object being made.
        :return: TF variable object.
        '''

        # Prep variable initializer function
        if self.yarotsky_initialization:
            initer = self.get_linker_weights_yarotsky_init(name)

            # Create variable for first weights in current neuron
            var = tf.get_variable(name=name, dtype=tf.float64, initializer=initer, trainable=self.trainable)
        else:
            initer = tf.contrib.layers.xavier_initializer()

            # Create variable for first weights in current neuron
            var = tf.get_variable(name=name, dtype=tf.float64, shape=[self.max_degree + 1, 1], initializer=initer, trainable=self.trainable)

        # Store in dictionary for saving trained net
        self.variable_dict[name] = var

        return var

    def get_linker_weights_yarotsky_init(self, name):
        '''
        Function used to create weights object for the linker nodes in case of Yarotsky initialization.

        :param name: The name of the current weights TF object being made.
        :return: TF constant object with values based on Yarotsky's formula.
        '''
        # Create np array which contains the weight to multiply the output of each net by
        linker_weights = []
        for hat_degree in range(self.max_degree + 1):
            # Deal with hat_funct_0(x) separately as it is the only positive weight
            if hat_degree == 0:
                linker_weights.append(1)
            # For all other degrees, weight is -2^(-2*hat_degree) - based on Yarotsky's article
            else:
                linker_weights.append(-2 ** (-2 * hat_degree))

        # Return weight object - constant column vector with max_degree + 1 values
        return tf.constant(value=np.array(linker_weights), shape=[self.max_degree+1 , 1], dtype=tf.float64, name=name)

#TESTING AREA#
if __name__ == '__main__':
    pass

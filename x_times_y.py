import tensorflow as tf
import numpy as np
from x_squared import XSquared

# CONSTANTS
from x_squared import USE_UNDYING_RELU

class XY():
    def __init__(self, max_degree, naming_postfix, input_placeholder_x=None, input_placeholder_y=None,
                 graph=None, yarotsky_initialization=False, ud_relus=USE_UNDYING_RELU, x_squared=None, y_squared=None,
                 tf_squaring_modules=False, trainable=True):
        self.max_degree = max_degree

        self.naming_postfix = naming_postfix

        self.tf_squaring_modules = tf_squaring_modules

        self.trainable = trainable

        if graph == None:
            self.tf_graph = tf.Graph()
        else:
            self.tf_graph = graph

        self.yarotsky_initialization = yarotsky_initialization

        self.use_ud_relus = ud_relus

        # Create/set the two x^2 blocks before linking them with the relevant weights
        if x_squared == None:
            self.x_squared = XSquared(self.max_degree, 'x_square'+naming_postfix+'_x', input_placeholder=input_placeholder_x,
                                 graph=self.tf_graph,yarotsky_initialization=self.yarotsky_initialization,
                                 ud_relus=self.use_ud_relus, trainable=trainable)
        else:
            self.x_squared = x_squared
        self.input_placeholder_x = self.x_squared.input_placeholder

        if y_squared == None:
            self.y_squared = XSquared(self.max_degree, 'y_square'+naming_postfix + '_y',input_placeholder=input_placeholder_y,
                                 graph=self.tf_graph, yarotsky_initialization=self.yarotsky_initialization,
                                 ud_relus=self.use_ud_relus, trainable=trainable)
        else:
            self.y_squared = y_squared
        self.input_placeholder_y = self.y_squared.input_placeholder


        with self.tf_graph.as_default():
            # Creation of (x+y)^2 block
            TEMP_x_plus_y_input = tf.add(self.input_placeholder_x, self.input_placeholder_y,
                                                name='input_adder_' + self.naming_postfix)
            x_plus_y_input = tf.matmul(TEMP_x_plus_y_input, tf.constant(value=np.array([0.5]), shape=[1, 1], dtype=tf.float64))

            self.x_plus_y_squared = XSquared(self.max_degree, 'x_plus_y'+self.naming_postfix,
                                             input_placeholder=x_plus_y_input, graph=self.tf_graph,
                                             yarotsky_initialization=self.yarotsky_initialization,
                                             ud_relus=self.use_ud_relus, trainable=trainable)

            mat_weights = self.get_linking_weights()
            if self.tf_squaring_modules:
                tf_x_squared = tf.math.square(self.input_placeholder_x)
                tf_y_squared = tf.math.square(self.input_placeholder_y)
                tf_x_plus_y_squared = tf.math.square(x_plus_y_input)
                squares_vector = tf.concat([tf_x_plus_y_squared, tf_x_squared,tf_y_squared],
                                           name='squares_outputs' + self.naming_postfix, axis=1)
            else:
                squares_vector = tf.concat([self.x_plus_y_squared.final_output, self.x_squared.final_output,
                                        self.y_squared.final_output], name='squares_outputs'+self.naming_postfix,axis=1)

            self.final_output = tf.matmul(squares_vector, mat_weights, name='matmult_linker'+self.naming_postfix)

    def get_linking_weights(self):

        # Prep variable initializer function
        if self.yarotsky_initialization:
            initer = self.get_linking_weights_yarotsky()
        else:
            initer = tf.contrib.layers.xavier_initializer()

        if self.yarotsky_initialization:
            var = tf.get_variable(name='monomial_linker'+self.naming_postfix, dtype=tf.float64, initializer=initer, trainable=self.trainable)
        else:
            var = tf.get_variable(name='monomial_linker'+self.naming_postfix, dtype=tf.float64, shape=[3, 1], initializer=initer, trainable=self.trainable)

        return var

    def get_linking_weights_yarotsky(self):
        return tf.constant(value=np.array([2, -0.5, -0.5]), shape=[3, 1], dtype=tf.float64)
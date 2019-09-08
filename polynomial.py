from x_squared import XSquared
from x_times_y import XY
import tensorflow as tf
import math

# CONSTANTS
from x_squared import USE_UNDYING_RELU

class Polynomial():
    def __init__(self, net_degree, naming_postfix, coefficients,input_placeholder=None,
                 graph=None, yarotsky_initialization=False, ud_relus=USE_UNDYING_RELU, tf_squaring_modules=False,
                 trainable=True):
        self.polynomial_degree = len(coefficients) - 1

        self.net_degree = net_degree

        self.naming_postfix = naming_postfix

        self.coefficients = coefficients

        self.monomial_outputs = []

        self.tf_squaring_modules = tf_squaring_modules

        self.trainable = trainable

        if graph == None:
            self.tf_graph = tf.Graph()
        else:
            self.tf_graph = graph

        self.yarotsky_initialization = yarotsky_initialization

        self.use_ud_relus = ud_relus

        if input_placeholder == None:
            with self.tf_graph.as_default():
                self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name=self.naming_postfix+'_input_placeholder')
        else:
            self.input_placeholder = input_placeholder

        self.build_net()


    def build_net(self):

        with self.tf_graph.as_default():
            # Iterate over necessary monomials, appending each level to the graph based on Yarotsky's architecture
            for monomial_degree in range(self.polynomial_degree+1):

                # Deal with constant (X^0)
                if monomial_degree == 0:
                    self.constant_tensor = tf.constant(value=self.coefficients[0], shape=[1, 1], dtype=tf.float32)
                    self.monomial_outputs.append(self.constant_tensor)

                # Deal with X^1
                elif monomial_degree == 1:
                    self.monomial_outputs.append(self.input_placeholder)

               # Deal with even polynomial degrees
                elif monomial_degree % 2 == 0:
                    if self.tf_squaring_modules:
                        self.monomial_outputs.append(tf.math.square(self.monomial_outputs[int(monomial_degree/2)]))
                    else:
                        squarer = XSquared(max_degree=self.net_degree, naming_postfix=self.naming_postfix+'_xsquared'+str(monomial_degree),
                                       input_placeholder=self.monomial_outputs[int(monomial_degree/2)], graph=self.tf_graph,
                                       yarotsky_initialization=self.yarotsky_initialization,ud_relus=self.use_ud_relus, trainable=self.trainable)
                        self.monomial_outputs.append(squarer.final_output)

                # Deal with odd polynomial degrees # TODO can also multiply x^[monomial_degree-1]*x
                elif monomial_degree % 2 == 1:
                    multiplier = XY(max_degree=self.net_degree, naming_postfix=self.naming_postfix+'xy'+str(monomial_degree),
                                    input_placeholder_x=self.monomial_outputs[int(monomial_degree/2)],
                                    input_placeholder_y=self.monomial_outputs[int(monomial_degree/2)+1],
                                    graph=self.tf_graph, yarotsky_initialization=self.yarotsky_initialization,
                                    ud_relus=USE_UNDYING_RELU, tf_squaring_modules=self.tf_squaring_modules, trainable=self.trainable)
                    self.monomial_outputs.append(multiplier.final_output)

            # Now "link" the monomials - multiply them by their corresponding coefficients and sum them up
            self.monomial_tensor = tf.concat(self.monomial_outputs[1:], name=self.naming_postfix+'_monomial_vector', axis=1)

            # Create coefficient tensor without constant (which will be added after)
            coefficient_tensor = tf.constant(value=self.coefficients[1:], shape=[len(self.coefficients)-1, 1], dtype=tf.float32)

            matmul = tf.matmul(self.monomial_tensor, coefficient_tensor, name=self.naming_postfix+'_output')

            self.final_output = tf.add(matmul, self.constant_tensor)
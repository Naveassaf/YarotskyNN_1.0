import tensorflow as tf
import numpy as np
from polynomial import Polynomial
import sympy as sy

# CONSTANTS
from x_squared import USE_UNDYING_RELU
RANDOM_COEFF_RANGE = (1, 3)

class AdaptivePolynomial(Polynomial):
    def __init__(self, naming_postfix, coefficients=[],input_placeholder=None,
                 graph=None, ud_relus=USE_UNDYING_RELU, polynomial_degree=None, sympy_expr=None):

        self.feed_dict = {}

        self.sympy_expr = sympy_expr

        if len(coefficients) == 0:
            # Create random coefficients for the polynomial
            coefficients = np.random.random(size=polynomial_degree) * (RANDOM_COEFF_RANGE[1] - RANDOM_COEFF_RANGE[0])\
                           + RANDOM_COEFF_RANGE[0]

        Polynomial.__init__(self, net_degree=None, naming_postfix=naming_postfix, coefficients=coefficients
                            ,input_placeholder=input_placeholder, graph=graph, yarotsky_initialization=False,
                            ud_relus=ud_relus, tf_squaring_modules=False, trainable=False)
    def build_net(self):

        with self.tf_graph.as_default():
            # Iterate over necessary monomials, appending each level to the graph based on Yarotsky's architecture
            for monomial_degree in range(self.polynomial_degree+1):

                # Deal with constant (X^0)
                if monomial_degree == 0:
                    self.constant_PH = tf.placeholder(shape=[1, 1], dtype=tf.float64)
                    self.feed_dict[self.constant_PH] = np.array([self.coefficients[0]]).reshape([1,1])
                    self.monomial_outputs.append(self.constant_PH)

                # Deal with X^1
                elif monomial_degree == 1:
                    self.monomial_outputs.append(self.input_placeholder)

                # Deal with all degrees larger than 1 (x^2, x^3, x^4, ...)
                else:
                    self.monomial_outputs.append(tf.math.pow(x=self.input_placeholder, y=float(monomial_degree)))

            # Now "link" the monomials - multiply them by their corresponding coefficients and sum them up
            self.monomial_tensor = tf.concat(self.monomial_outputs[1:], name=self.naming_postfix+'_monomial_vector', axis=1)

            # Create coefficient tensor without constant (which will be added after)
            self.coeffs_PH = tf.placeholder(shape=[len(self.coefficients) - 1, 1], dtype=tf.float64)
            self.feed_dict[self.coeffs_PH] = np.array(self.coefficients[1:]).reshape([len(self.coefficients) - 1, 1])

            matmul = tf.matmul(self.monomial_tensor, self.coeffs_PH, name=self.naming_postfix+'_output')

            self.final_output = tf.add(matmul, self.constant_PH)

    def get_updated_feed_dict(self, taylor_x0):
        # Recalculate Taylor coefficients
        new_coeffs = self.get_taylor_coeffs(taylor_x0=taylor_x0)
        print('New Coeffs: {}'.format(new_coeffs))

        self.feed_dict[self.constant_PH] = np.array([new_coeffs[0]]).reshape([1,1])
        self.feed_dict[self.coeffs_PH] = np.array(new_coeffs[1:]).reshape([len(new_coeffs) - 1, 1])
        return self.feed_dict

    def get_taylor_coeffs(self, taylor_x0):
        # Create Polynomial Sympy object with the Taylor coefficients corresponding to parsed expression
        sympy_poly = sy.Poly(sy.series(expr=self.sympy_expr, x=sy.Symbol('x'), n=self.polynomial_degree+1,
                                       x0=float(taylor_x0)).removeO())

        # Return the calculated coefficients (must reverse order to match implementation)
        coeffs = sympy_poly.all_coeffs()
        coeffs = np.array(coeffs).astype(np.float64)
        return coeffs[::-1]

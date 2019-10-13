from polynomial import Polynomial
from x_squared import XSquared
from x_times_y import XY
from bump import Bump
from net_master import NetMaster
from tf_polynomial import TFPolynomial
from rigid_bump import RigidBump
from adaptive_polynomial import AdaptivePolynomial

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sympy.parsing.sympy_parser as parser
import sympy as sy

# TODO: COMMENT OUT FOR WINDOWS (questionable patch for MacOS)
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# TODO: END

# CONSTANTS
func_expression = ''
sy_x = sy.Symbol('x')
DEFAULT_POLY_DEGREE = 10

def func(x):
    '''
    This function serves the NetMaster in the generation of the (x,y) pairs which the net uses to train, test,
    and validate. It is passed as an object to the NetMaster's constructor.

    :param x: The sampling point/s. May be int/float or a np.array
    :return: Returns f(x) (vector or scalar, corresponding to input x)
    '''
    global func_expression

    ### MAKE SURE EXPRESSION AND FUNCTION MATCH! ###
    func_expression = 'sin(10*x)+cos(15*x)-x**2'
    return np.sin(10*x)+np.cos(15*x)-x**2
    # func_expression = '0.1+0.2*x+0.3*x**2+0.4*x**3+0.1*x**4'
    # return 0.1+0.2*x+0.3*x**2+0.4*x**3+0.1*x**4
    # func_expression = '0.0001*x+1'
    # return 0.0001*x+1
    ################################################

def example_x_squared():
    '''
    Example code of how to create two x^2 blocks, link them by using a matmul tensor and evaluate the final graph's
    output.

    :return:
    '''
    # List to store all blocks' outputs
    block_outputs = []

    # Create two x^2 blocks and append their outputs to block_output list
    block1 = XSquared(7, naming_postfix='first', yarotsky_initialization=True)
    block2 = XSquared(7, naming_postfix='second', graph=block1.tf_graph, input_placeholder=block1.input_placeholder,
                      yarotsky_initialization=True)
    block_outputs.append(block1.final_output)
    block_outputs.append(block2.final_output)

    # Using matmul, add outputs of the two x^2 blocks, multiplying by 2, 1 respectively
    with block1.tf_graph.as_default():
        # Create output placeholder to be passed to the NetMaster for training
        output_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

        # Concatenate the outputs into one tensor. Create a vecotr to multiply them by and get the final output
        linker = tf.concat(block_outputs, name='block_outputs', axis=1)
        final_output = tf.matmul(linker, tf.constant(value=np.array([2.0, 1.0]), shape=[2, 1], dtype=tf.float64))

    # Create NetMaster module to evaluate the created graph. Evaluate at point x (input parameter
    master = NetMaster(tf_graph=block1.tf_graph, function=func, net_output=final_output,
                       input_placeholder=block1.input_placeholder, output_placeholder=output_placeholder,
                       sampling_resolution=1e-6, learning_rate=1e-3)

    # Test before and plot after training
    inputs = np.arange(0,1.01,0.01)
    plt.plot(inputs, master.evaluate(inputs), color='g')
    master.train()
    plt.plot(inputs, master.evaluate(inputs))
    plt.show()

def example_x_times_y():
    '''
    Example of using the XY module. This function creates two input placeholders and multiplies them using an XY net.

    :return:
    '''
    # Create graph to which XY components (including passed input placeholders) are passed.
    graph = tf.Graph()
    with graph.as_default():
        # Create input place holder - "x" value
        x_input_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')

        # Add some const. to X to create Y.
        y_input_placeholder = tf.add(x_input_placeholder, tf.constant(value=0.2, shape=[1, 1], dtype=tf.float64))

        # Prep output placeholder for training
        output_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    # Create xy object
    xy = XY(4, 'tester',input_placeholder_x=x_input_placeholder, input_placeholder_y=y_input_placeholder, graph=graph,
            yarotsky_initialization=False)

    # Create master to train and evaluate net.
    master = NetMaster(tf_graph=xy.tf_graph, function=func, net_output=xy.final_output,
                       input_placeholder=x_input_placeholder, output_placeholder=output_placeholder,
                       sampling_resolution=1e-6, learning_rate=1e-3)

    # Test for x=0.3, y=0.5
    inputs = 0.3
    graphs_xs = np.arange(0, 1.01, 0.01)

    # MSE before training
    print(master.calc_mse(func))

    feed_dict = {}
    feed_dict[x_input_placeholder] = np.array(inputs, dtype='float64').reshape((1, 1))
    plt.plot(graphs_xs, master.evaluate(graphs_xs), color = 'r') # Plot before training
    # print(master.sess.run([x_input_placeholder, y_input_placeholder], feed_dict=feed_dict))
    master.train()

    # MSE after training
    print(master.calc_mse(func))

    # Plot before training
    plt.plot(graphs_xs, master.evaluate(graphs_xs), color='g')
    plt.show()

def example_polynomial(coefficients, net_degree=10, learning_rate=1e-3, sampling_resolution=1e-7, yarotsky_init=True,
                       save_path=None, sympy_poly=None, tf_squaring_modules=False, trainable=True):
    '''
    Function of creating a polynomial NeuralNetwork given the polynomial's coefficients.

    :param coefficients: Polynomial's coefficients. Poly = coef[0] + coef[1]*X+ coef[2]*X^2 + coef[3]*X^3...
    :return:
    '''

    # Create polynomial with the provided argumnets...
    poly = Polynomial(net_degree=net_degree, naming_postfix='tester', coefficients=coefficients,yarotsky_initialization=yarotsky_init
                      ,ud_relus=not yarotsky_init, tf_squaring_modules=tf_squaring_modules, trainable=trainable)

    # Create output placehoder to be used by master for training.
    with poly.tf_graph.as_default():
        output_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    # Create master which will train and evaluate polynomial network.
    master = NetMaster(tf_graph=poly.tf_graph, function=func,net_output=poly.final_output,
                       input_placeholder=poly.input_placeholder, output_placeholder=output_placeholder,
                       sampling_resolution=sampling_resolution, learning_rate=learning_rate, trainable_net=trainable)

    # Graph polynomial function
    inputs = np.arange(0, 1, 0.001)
    out = master.evaluate(inputs)

    initial_mse = master.calc_mse(func)

    plt.figure(1)
    if not sympy_poly == None:
        plt.plot(inputs, eval_sympy_polynomial_vector(inputs, sympy_poly), color='r')
    plt.plot(inputs, out, color='g')
    plt.plot(inputs, func(inputs), color='b')

    # Train and reevaluate network
    if trainable:
        master.train()

    out = master.evaluate(inputs)
    plt.plot(inputs, out, color='c')

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    # Plot convergence of net
    plt.figure(2)
    plt.plot(range(len(master.loss_list)), master.loss_list)
    plt.yscale('log')

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path.replace('.pdf', '_LOSS.pdf'))
        plt.clf()

    final_mse = master.calc_mse(func)

    return initial_mse,final_mse


def example_taylor(taylor_x0, polynomial_degree=DEFAULT_POLY_DEGREE, print_to_console=False):
    '''
    An example function of generating the Taylor series coefficients of the function provided in func() using the
    Sympy library

    :param taylor_x0: Point around which Taylor Series is calculated
    :param polynomial_degree: The degree to which the Taylor Series is calculated. Takes global default if not provided.
    :return: List of Taylor Series coefficients. coefficients[i] is the coefficient of X^i
    '''
    # Ensures function expression has been set
    func(1)

    # Use the Sympy library to parse the function_expression set in func() - global variable altered there
    sympy_expr = parser.parse_expr(func_expression, evaluate=True,
                          transformations=parser.standard_transformations + (parser.implicit_multiplication,))

    # Create Polynomial Sympy object with the Taylor coefficients corresponding to parsed expression
    sympy_poly = sy.Poly(sy.series(expr=sympy_expr, x=sy_x,n=polynomial_degree,x0=float(taylor_x0)).removeO())

    if print_to_console:
        # Give a nice little status print
        print('--------------------------------------------\n'+
              'Taylor Degree: {}\n\n X0: {}\n\nPolynomial: {}\n\nCoefficients: {}\n\nP(0.8) = {}\n\nf(0.8) = {}\n\n'.format(
                    polynomial_degree, taylor_x0,sympy_poly,sympy_poly.all_coeffs(), sympy_poly.eval(0.8), func(0.8))+
            '--------------------------------------------\n')

    # Return the calculated coefficients
    coeffs = sympy_poly.all_coeffs()
    coeffs = np.array(coeffs).astype(np.float64)
    return [coeffs[::-1], sympy_poly]

def example_bumps():
    ''''
    Example of creating and ploting bump functions for N = 3
    '''

    bump0 = Bump(3, 0, '0_bump',  yarotsky_initialization=True, ud_relus=False)
    bump1 = Bump(3, 1, '1_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder,yarotsky_initialization=True, ud_relus=False)
    bump2 = Bump(3, 2, '2_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder,yarotsky_initialization=True, ud_relus=False)
    bump3 = Bump(3, 3, '3_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder,yarotsky_initialization=True, ud_relus=False)

    # Create output placehoder to be used by master for training.
    with bump0.tf_graph.as_default():
        output_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    # Create master which will train and evaluate polynomial network.
    master = NetMaster(tf_graph=bump0.tf_graph, function=func, net_output=bump1.final_output,
                       input_placeholder=bump0.input_placeholder, output_placeholder=output_placeholder,
                       sampling_resolution=3e-8, learning_rate=1e-3)

    # Graph polynomial function
    inputs = np.arange(0, 1, 0.001)
    plt.figure(1)
    out = master.evaluate(inputs, node=bump0.final_output)
    plt.plot(inputs, out, color='r')
    out = master.evaluate(inputs, node=bump1.final_output)
    plt.plot(inputs, out, color='g')
    out = master.evaluate(inputs, node=bump2.final_output)
    plt.plot(inputs, out, color='b')
    out = master.evaluate(inputs, node=bump3.final_output)
    plt.plot(inputs, out, color='y')

    # Train second Bump to func() and plot again
    master.train()
    plt.figure(2)
    out = master.evaluate(inputs, node=bump0.final_output)
    plt.plot(inputs, out, color='r')
    out = master.evaluate(inputs, node=bump1.final_output)
    plt.plot(inputs, out, color='g')
    out = master.evaluate(inputs, node=bump2.final_output)
    plt.plot(inputs, out, color='b')
    out = master.evaluate(inputs, node=bump3.final_output)
    plt.plot(inputs, out, color='y')
    plt.show()

def example_rigid_bumps():
    ''''
    Example of creating and ploting bump functions for N = 3
    '''

    bump0 = RigidBump(3, 0, '0_bump',  ud_relus=False)
    bump1 = RigidBump(3, 1, '1_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder, ud_relus=False)
    bump2 = RigidBump(3, 2, '2_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder, ud_relus=False)
    bump3 = RigidBump(3, 3, '3_bump', graph=bump0.tf_graph, input_placeholder=bump0.input_placeholder, ud_relus=False)

    # Create output placehoder to be used by master for training.
    with bump0.tf_graph.as_default():
        output_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    # Create master which will train and evaluate polynomial network.
    master = NetMaster(tf_graph=bump0.tf_graph, function=func, net_output=bump2.final_output,
                       input_placeholder=bump0.input_placeholder, output_placeholder=output_placeholder,
                       sampling_resolution=1e-7, learning_rate=1e-3)


    # Graph polynomial function
    inputs = np.arange(0, 1, 0.001)
    plt.figure(1)
    out = master.evaluate(inputs, node=bump0.final_output)
    plt.plot(inputs, out, color='r')
    out = master.evaluate(inputs, node=bump1.final_output)
    plt.plot(inputs, out, color='g')
    out = master.evaluate(inputs, node=bump2.final_output)
    plt.plot(inputs, out, color='b')
    out = master.evaluate(inputs, node=bump3.final_output)
    plt.plot(inputs, out, color='y')


    # Train second Bump to func() and plot again. Check that only first 3 elements are updated
    master.train()

    plt.figure(2)
    out = master.evaluate(inputs, node=bump0.final_output)
    plt.plot(inputs, out, color='r')
    out = master.evaluate(inputs, node=bump1.final_output)
    plt.plot(inputs, out, color='g')
    out = master.evaluate(inputs, node=bump2.final_output)
    plt.plot(inputs, out, color='b')
    out = master.evaluate(inputs, node=bump3.final_output)
    plt.plot(inputs, out, color='y')
    plt.plot(inputs, func(inputs), color='m')
    plt.show()

def spline_polynomial(N, taylor_degree, composite_degree, save_path=None, sampling_res=1e-7, learning_rate=1e-3,
                      yar_init=True, tf_squaring_modules=False, train_polynomial=True, train_bumps=True):
    tf_graph = tf.Graph()

    with tf_graph.as_default():
        input_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
        output_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    sympy_taylor_dict = {}

    poly_dict = {}
    poly_outputs = []

    bump_dict = {}
    bump_outputs = []

    for index in range(N+1):
        bump_dict[index] = Bump(N, index, naming_postfix='bump'+str(index), graph=tf_graph, input_placeholder=input_PH,
                                yarotsky_initialization=True, ud_relus=False, trainable=train_bumps)
        bump_outputs.append(bump_dict[index].final_output)

        current_coeffs, sympy_taylor_dict[index] = example_taylor(taylor_x0=bump_dict[index].bump_center, polynomial_degree=taylor_degree)

        print('index: {}, around: {}, poly {}'.format(index,bump_dict[index].bump_center, sympy_taylor_dict[index]))

        poly_dict[index] = Polynomial(net_degree=composite_degree, naming_postfix='poly'+str(index), coefficients=current_coeffs,
                                      yarotsky_initialization=yar_init, ud_relus=not yar_init, input_placeholder=input_PH,
                                      graph=tf_graph, tf_squaring_modules=tf_squaring_modules, trainable=train_polynomial)
        poly_outputs.append(poly_dict[index].final_output)

    bump_vector = tf.concat(bump_outputs, name='bump_vector', axis=1)
    poly_vector = tf.concat(poly_outputs, name='poly_vector', axis=1)
    with tf_graph.as_default():
        temp = tf.multiply(bump_vector, poly_vector)
        net_output = tf.matmul(temp, tf.constant(value=np.ones([N+1, 1]), shape=[N+1, 1], dtype=tf.float64))

    # Create master which will train and evaluate spline polynomial network.
    master = NetMaster(tf_graph=tf_graph, function=func, net_output=net_output,
                           input_placeholder=input_PH, output_placeholder=output_PH,
                           sampling_resolution=sampling_res, learning_rate=learning_rate, trainable_net=train_polynomial)

    mse_before_training = master.calc_mse(func)
    plt.figure(1)
    # Plotting
    range_start = 0
    half_range_width = 1.0/(2*float(N))
    for index in range(N+1):
        if index == 0:
            inputs = np.arange(0,half_range_width, 0.001)
            range_start = half_range_width
        elif index == N:
            inputs = np.arange(range_start, 1.001, 0.001)
        else:
            inputs = np.arange(range_start, range_start+2*half_range_width, 0.001)
            range_start += 2*half_range_width

        plt.plot(inputs, eval_sympy_polynomial_vector(inputs, sympy_taylor_dict[index]), color = 'r', linewidth=1.0)
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color = 'y', linewidth=1.0)

    inputs = np.arange(0, 1.001, 0.001)
    plt.plot(inputs, func(inputs), color = 'b', linewidth=1.0)

    if mse_before_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color = 'g', linewidth=1.0)

    master.train(print_to_console=True)


    #TODO ERASE:
    for index in range(N+1):
        if index == 0:
            inputs = np.arange(0,half_range_width, 0.001)
            range_start = half_range_width
        elif index == N:
            inputs = np.arange(range_start, 1.001, 0.001)
        else:
            inputs = np.arange(range_start, range_start+2*half_range_width, 0.001)
            range_start += 2*half_range_width

        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color = 'm', linewidth=1.0)
    #TODO: END

    mse_after_training = master.calc_mse(func)
    if mse_after_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color = 'c', linewidth=1.0)

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    plt.figure(2)
    plt.plot(range(len(master.loss_list)), master.loss_list)
    plt.yscale('log')
    if not save_path == None:
        plt.savefig(save_path.replace('.pdf', '_LOSS.pdf'))

    plt.clf()

    return mse_before_training, mse_after_training

def spline_tf_polynomial(N, taylor_degree, save_path=None, sampling_res=1e-7, learning_rate=1e-3,
                      ud_relus=False, train_polynomial=True, train_bumps=True, taylor_init=True):

    tf_graph = tf.Graph()

    with tf_graph.as_default():
        input_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
        output_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    sympy_taylor_dict = {}

    poly_dict = {}
    poly_outputs = []

    bump_dict = {}
    bump_outputs = []

    for index in range(N+1):
        bump_dict[index] = Bump(N, index, naming_postfix='bump'+str(index), graph=tf_graph, input_placeholder=input_PH,
                                ud_relus=False, yarotsky_initialization=True, trainable=train_bumps)
        bump_outputs.append(bump_dict[index].final_output)


        current_coeffs, sympy_taylor_dict[index] = example_taylor(taylor_x0=bump_dict[index].bump_center, polynomial_degree=taylor_degree)

        print('index: {}, around: {}, poly {}'.format(index,bump_dict[index].bump_center, sympy_taylor_dict[index]))

        if not taylor_init:
            poly_dict[index] = TFPolynomial(naming_postfix='poly'+str(index), coefficients=[],
                                            ud_relus=ud_relus, input_placeholder=input_PH, graph=tf_graph, trainable=train_polynomial,
                                        polynomial_degree=len(current_coeffs))
        else:
            poly_dict[index] = TFPolynomial(naming_postfix='poly' + str(index), coefficients=current_coeffs,
                                            ud_relus=ud_relus, input_placeholder=input_PH, graph=tf_graph, trainable=train_polynomial)

        poly_outputs.append(poly_dict[index].final_output)

    bump_vector = tf.concat(bump_outputs, name='bump_vector', axis=1)
    poly_vector = tf.concat(poly_outputs, name='poly_vector', axis=1)
    with tf_graph.as_default():
        temp = tf.multiply(bump_vector, poly_vector)
        net_output = tf.matmul(temp, tf.constant(value=np.ones([N+1, 1]), shape=[N+1, 1], dtype=tf.float64))

    # Create master which will train and evaluate spline polynomial network.
    master = NetMaster(tf_graph=tf_graph, function=func, net_output=net_output,
                           input_placeholder=input_PH, output_placeholder=output_PH,
                           sampling_resolution=sampling_res, learning_rate=learning_rate, trainable_net=train_polynomial or train_bumps)

    mse_before_training = master.calc_mse(func)
    plt.figure(1)
    # Plotting
    range_start = 0
    half_range_width = 1.0/(2*float(N))
    for index in range(N+1):
        if index == 0:
            inputs = np.arange(0,half_range_width, 0.001)
            range_start = half_range_width
        elif index == N:
            inputs = np.arange(range_start, 1.001, 0.001)
        else:
            inputs = np.arange(range_start, range_start+2*half_range_width, 0.001)
            range_start += 2*half_range_width

        plt.plot(inputs, eval_sympy_polynomial_vector(inputs, sympy_taylor_dict[index]), color = 'r', linewidth=1.0)
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color = 'y', linewidth=1.0)

    inputs = np.arange(0, 1.001, 0.001)
    plt.plot(inputs, func(inputs), color = 'b', linewidth=1.0)

    if mse_before_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color = 'g', linewidth=1.0)

    master.train(print_to_console=True)


    #TODO ERASE:
    for index in range(N+1):
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color = 'm', linewidth=1.0)
    #TODO: END

    mse_after_training = master.calc_mse(func)
    if mse_after_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color = 'c', linewidth=1.0)

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    plt.figure(2)
    plt.plot(range(len(master.loss_list)), master.loss_list)
    plt.yscale('log')
    if not save_path == None:
        plt.savefig(save_path.replace('.pdf', '_LOSS.pdf'))

    plt.clf()

    return mse_before_training, mse_after_training

def eval_sympy_polynomial_vector(input_array, sympy_polynomial):
    poly_outputs = []
    for x in input_array:
        poly_outputs.append(sympy_polynomial.eval(x))
    return poly_outputs

def spline_automation(output_dir):
    counter = 0
    N_list = [3,5,7]
    yarotsky_degree_list = [3,5,7]
    learning_rates = [1e-3, 1e-4, 1e-5]
    taylor_remainder_degree_list = [2,4,6]
    yarotsky_init_list = [True, False]

    with open(output_dir+'\\summary.txt', 'w') as output_fp:
        for yarotsky_init in yarotsky_init_list:
            for N in N_list:
                for yarotsky_degree, learning_rate in zip(yarotsky_degree_list, learning_rates):
                    for taylor_remainder_degree in taylor_remainder_degree_list:
                        if not yarotsky_init:
                            learning_rate = 1e-3
                        counter+=1
                        try:
                            outname = ('YarInit_{}___N_{}___YarDeg_{}___RemDeg_{}.pdf').format(yarotsky_init, N, yarotsky_degree,
                                                                        taylor_remainder_degree)
                            init_mse, final_mse = spline_polynomial(N=N, taylor_degree=taylor_remainder_degree,
                                                                    composite_degree=yarotsky_degree, sampling_res=3e-8,
                                                                    learning_rate=learning_rate,yar_init=yarotsky_init ,save_path=output_dir+'\\'+outname)
                            print('\n\n\n\n\nNet Number {} - SUCCESS\n\n\n\n\n'.format(counter))
                        except:
                            init_mse, final_mse = ['FAILED', 'FAILED']
                            print('\n\n\n\n\nNet Number {} - FAILURE\n\n\n\n\n'.format(counter))
                        output_fp.write('oOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOo\n\n')
                        output_fp.write('Yarotsky Init = {}\nN = {}\nYarotsky Composition Degree = {} --> Learning Rate = {}\n'
                                        'Taylor Error Degree = {}\nInitial MSE = {}\nFinal MSE = {}\n\n'.format(
                            yarotsky_init, N, yarotsky_degree, learning_rate,taylor_remainder_degree, init_mse, final_mse))
                        output_fp.write('oOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOo\n\n')

def normal_polynomial_automation(outdir):
    counter = 0
    yarotsky_degree_list = [3, 7]
    learning_rates = [1e-3, 1e-5]
    taylor_remainder_degree_list = [4, 6]
    yarotsky_init_list = [True, False]

    with open(outdir + '\\summary.txt', 'w') as output_fp:
        for yarotsky_init in yarotsky_init_list:
            for yarotsky_degree, learning_rate in zip(yarotsky_degree_list, learning_rates):
                for taylor_remainder_degree in taylor_remainder_degree_list:
                    if not yarotsky_init:
                        learning_rate = 1e-3
                    counter += 1



                    try:
                        outname = ('YarInit_{}___YarDeg_{}___RemDeg_{}.pdf').format(yarotsky_init,
                                                                                               yarotsky_degree,
                                                                                               taylor_remainder_degree)
                        coeffs, taylor_poly = example_taylor(taylor_x0=0.5, polynomial_degree=taylor_remainder_degree, print_to_console=False)
                        init_mse, final_mse = example_polynomial(coeffs, net_degree=yarotsky_degree, learning_rate=learning_rate, sampling_resolution=3e-8
                                           ,yarotsky_init=yarotsky_init, save_path=outdir+'\\'+outname, sympy_poly=taylor_poly)
                        print('\n\n\n\n\nNet Number {} - SUCCESS\n\n\n\n\n'.format(counter))
                    except:
                        init_mse, final_mse = ['FAILED', 'FAILED']
                        print('\n\n\n\n\nNet Number {} - FAILURE\n\n\n\n\n'.format(counter))





                    output_fp.write('oOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOo\n\n')
                    output_fp.write(
                            'Yarotsky Init = {}\nYarotsky Composition Degree = {} --> Learning Rate = {}\n'
                            'Taylor Error Degree = {}\nInitial MSE = {}\nFinal MSE = {}\n\n'.format(
                                yarotsky_init, yarotsky_degree, learning_rate, taylor_remainder_degree, init_mse,
                                final_mse))
                    output_fp.write('oOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOoOo\n\n')

def yarotsky_init_debug():
    mse_before_training, mse_after_training = spline_polynomial(N=5, taylor_degree=8, composite_degree=8,
                                                                learning_rate=1e-5, sampling_res=1e-6)
    print('MSE before training = ' + str(mse_before_training))
    mse_before_training, mse_after_training = spline_polynomial(N=5, taylor_degree=8, composite_degree=15,
                                                                learning_rate=1e-5, sampling_res=1e-6)
    print('MSE before training = ' + str(mse_before_training))
    mse_before_training, mse_after_training = spline_polynomial(N=5, taylor_degree=8, composite_degree=3,
                                                                learning_rate=1e-5, sampling_res=1e-6,
                                                                tf_squaring_modules=True)
    print('MSE before training = ' + str(mse_before_training))


def section_8A(N, taylor_degree, save_path=None, sampling_res=1e-7, learning_rate=1e-3,
                      ud_relus=False, train_polynomial=True, train_bumps=True, taylor_init=True):
    tf_graph = tf.Graph()

    with tf_graph.as_default():
        input_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
        output_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    sympy_taylor_dict = {}

    poly_dict = {}
    poly_outputs = []

    bump_dict = {}
    bump_outputs = []

    for index in range(N + 1):
        bump_dict[index] = RigidBump(N, index, naming_postfix='bump' + str(index), graph=tf_graph,
                                input_placeholder=input_PH, ud_relus=False)
        bump_outputs.append(bump_dict[index].final_output)


        current_coeffs, sympy_taylor_dict[index] = example_taylor(taylor_x0=bump_dict[index].bump_center,
                                                                  polynomial_degree=taylor_degree)

        print('index: {}, around: {}, poly {}'.format(index, bump_dict[index].bump_center, sympy_taylor_dict[index]))

        if taylor_init:
            poly_dict[index] = TFPolynomial(naming_postfix='poly' + str(index), coefficients=current_coeffs,
                                                ud_relus=ud_relus, input_placeholder=input_PH, graph=tf_graph,
                                                trainable=train_polynomial)
        else:
            poly_dict[index] = TFPolynomial(naming_postfix='poly' + str(index), coefficients=[],
                                            ud_relus=ud_relus, input_placeholder=input_PH, graph=tf_graph,
                                            trainable=train_polynomial, polynomial_degree=len(current_coeffs))


        poly_outputs.append(poly_dict[index].final_output)

    bump_vector = tf.concat(bump_outputs, name='bump_vector', axis=1)
    poly_vector = tf.concat(poly_outputs, name='poly_vector', axis=1)
    with tf_graph.as_default():
        temp = tf.multiply(bump_vector, poly_vector)
        net_output = tf.matmul(temp, tf.constant(value=np.ones([N + 1, 1]), shape=[N + 1, 1], dtype=tf.float64))

    # Create master which will train and evaluate spline polynomial network.
    master = NetMaster(tf_graph=tf_graph, function=func, net_output=net_output,
                       input_placeholder=input_PH, output_placeholder=output_PH,
                       sampling_resolution=sampling_res, learning_rate=learning_rate,
                       trainable_net=train_polynomial or train_bumps)

    mse_before_training = master.calc_mse(func)
    plt.figure(1)

    # Plotting
    range_start = 0
    half_range_width = 1.0 / (2 * float(N))
    whole_range_inputs = np.arange(0, 1.001, 0.001)
    for index in range(N + 1):
        if index == 0:
            inputs = np.arange(0, half_range_width, 0.001)
            range_start = half_range_width
        elif index == N:
            inputs = np.arange(range_start, 1.001, 0.001)
        else:
            inputs = np.arange(range_start, range_start + 2 * half_range_width, 0.001)
            range_start += 2 * half_range_width

        plt.plot(inputs, eval_sympy_polynomial_vector(inputs, sympy_taylor_dict[index]), color='r', linewidth=1.0)
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color='y', linewidth=1.0)

    inputs = np.arange(0, 1.001, 0.001)
    plt.plot(inputs, func(inputs), color='b', linewidth=1.0)

    if mse_before_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color='g', linewidth=1.0)

    master.train(print_to_console=True)

    for index in range(N + 1):
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color='m', linewidth=1.0)

    mse_after_training = master.calc_mse(func)
    if mse_after_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color='c', linewidth=1.0)

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    plt.figure(2)
    plt.plot(range(len(master.loss_list)), master.loss_list)
    plt.yscale('log')
    if not save_path == None:
        plt.savefig(save_path.replace('.pdf', '_LOSS.pdf'))

    plt.clf()

    return mse_before_training, mse_after_training

def section_8B(N, taylor_degree, save_path=None, sampling_res=1e-7, learning_rate=1e-3,
                      ud_relus=False):
    tf_graph = tf.Graph()

    with tf_graph.as_default():
        input_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='input_placeholder')
        output_PH = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='output_placeholder')

    sympy_taylor_dict = {}

    poly_dict = {}
    poly_outputs = []

    bump_dict = {}
    bump_outputs = []

    poly_list = []
    bump_list = []

    if func_expression == '':
        func(1)

    sympy_expr = parser.parse_expr(func_expression, evaluate=True, transformations=parser.standard_transformations + (parser.implicit_multiplication,))

    for index in range(N + 1):
        bump_dict[index] = RigidBump(N, index, naming_postfix='bump' + str(index), graph=tf_graph,
                                input_placeholder=input_PH, ud_relus=False)
        bump_outputs.append(bump_dict[index].final_output)
        bump_list.append(bump_dict[index])

        current_coeffs, sympy_taylor_dict[index] = example_taylor(taylor_x0=bump_dict[index].bump_center,
                                                                  polynomial_degree=taylor_degree)

        print('index: {}, around: {}, poly {}'.format(index, bump_dict[index].bump_center, sympy_taylor_dict[index]))

        poly_dict[index] = AdaptivePolynomial(naming_postfix='poly' + str(index), coefficients=current_coeffs,
                                                ud_relus=ud_relus, input_placeholder=input_PH, graph=tf_graph,
                                              sympy_expr=sympy_expr)

        poly_list.append(poly_dict[index])
        poly_outputs.append(poly_dict[index].final_output)

    bump_vector = tf.concat(bump_outputs, name='bump_vector', axis=1)
    poly_vector = tf.concat(poly_outputs, name='poly_vector', axis=1)
    with tf_graph.as_default():
        temp = tf.multiply(bump_vector, poly_vector)
        net_output = tf.matmul(temp, tf.constant(value=np.ones([N + 1, 1]), shape=[N + 1, 1], dtype=tf.float64))

    # Create master which will train and evaluate spline polynomial network.
    master = NetMaster(tf_graph=tf_graph, function=func, net_output=net_output,
                       input_placeholder=input_PH, output_placeholder=output_PH,
                       sampling_resolution=sampling_res, learning_rate=learning_rate,
                       trainable_net=True, adaptive_mode=True, poly_list=poly_list, bump_list=bump_list)

    mse_before_training = master.calc_mse(func)
    plt.figure(1)

    # Draw hat and taylor values before training
    range_start = 0
    half_range_width = 1.0 / (2 * float(N))
    whole_range_inputs = np.arange(0, 1.001, 0.001)
    for index in range(N + 1):
        if index == 0:
            inputs = np.arange(0, half_range_width, 0.001)
            range_start = half_range_width
        elif index == N:
            inputs = np.arange(range_start, 1.001, 0.001)
        else:
            inputs = np.arange(range_start, range_start + 2 * half_range_width, 0.001)
            range_start += 2 * half_range_width

        plt.plot(inputs, eval_sympy_polynomial_vector(inputs, sympy_taylor_dict[index]), color='r', linewidth=1.0)
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color='r', linewidth=1.0)

    # Draw function being estimated/learnt
    inputs = np.arange(0, 1.001, 0.001)
    plt.plot(inputs, func(inputs), color='b', linewidth=1.0)

    # Draw net's initial value
    if mse_before_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color='g', linewidth=1.0)

    # Train net
    master.train(print_to_console=True)

    # Draw new hats and taylor values after training
    for index in range(N + 1):
        plt.plot(inputs, master.evaluate(inputs, bump_dict[index].final_output), color='m', linewidth=1.0)

    # Plot net value after training
    mse_after_training = master.calc_mse(func)
    if mse_after_training < 10:
        plt.plot(inputs, master.evaluate(inputs), color='c', linewidth=1.0)

    # Save output if desired
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.clf()

    # Plot loss
    plt.figure(2)
    plt.plot(range(len(master.loss_list)), master.loss_list)
    plt.yscale('log')
    if not save_path == None:
        plt.savefig(save_path.replace('.pdf', '_LOSS.pdf'))

    plt.clf()

    return mse_before_training, mse_after_training


if __name__ == '__main__':

    '''BASIC EXAMPLES'''
    #--------------------------------------------------
    # ~~ X SQUARED
    # example_x_squared()


    # --------------------------------------------------
    # ~~ X TIMES Y
    # example_x_times_y()


    # --------------------------------------------------
    # ~~ NORMAL POLYNOMIAL
    # bef, aft = example_polynomial([0,0,0,0,0,0,0,1], sampling_resolution=1e-7,
    #                    learning_rate=1e-6, net_degree=3, trainable=True)


    # --------------------------------------------------
    # ~~ TAYLOR INITIALIZED POLYNOMIAL
    # coeffs,taylor_poly = example_taylor(taylor_x0=1, polynomial_degree=8, print_to_console=True)
    # example_polynomial(coeffs, sampling_resolution=1e-7, sympy_poly=taylor_poly, net_degree=6, tf_squaring_modules=True,
    #                    trainable=False)


    # --------------------------------------------------
    # ~~ SPLINE BASED ON FUNCTION
    # mse_before_training, mse_after_training = spline_polynomial(N=3, taylor_degree=4, composite_degree=6,
    #                                                             sampling_res=1e-6,learning_rate=1e-3,
    #                                                             train_polynomial=True, train_bumps=False)
    # print("Init MSE = {}, Final MSE = {}".format(mse_before_training, mse_after_training))
    # mse_before_training, mse_after_training = spline_polynomial(N=3, taylor_degree=4, composite_degree=6,
    #                                                             sampling_res=1e-6,learning_rate=1e-3,
    #                                                             train_polynomial=False, train_bumps=True)
    # print("Init MSE = {}, Final MSE = {}".format(mse_before_training, mse_after_training))


    # --------------------------------------------------
    # ~~ BUMPS
    # example_bumps()

    # --------------------------------------------------
    # ~~ POWER SPLINE BASED ON FUNCTION
    # mse_before_training, mse_after_training = spline_tf_polynomial(N=3, taylor_degree=4, sampling_res=1e-7,learning_rate=1e-3,
    #                                                                   ud_relus=False, train_polynomial=False, train_bumps=True)
    # print("Init MSE = {}, Final MSE = {}".format(mse_before_training, mse_after_training))
    #
    # mse_before_training, mse_after_training = spline_tf_polynomial(N=3, taylor_degree=4, sampling_res=5e-6,learning_rate=1e-3,
    #                                                                   ud_relus=False, train_polynomial=True, train_bumps=True,
    #                                                                taylor_init=False)
    # print("Init MSE = {}, Final MSE = {}".format(mse_before_training, mse_after_training))


    # -------------------------------------------------
    # ~~ RIGID BUMPS
    # example_rigid_bumps()


    '''AUTOMATIONS'''
    # spline_automation('C:\\Users\\navea\\Desktop\\yarotsky_automation_spline')
    # --------------------------------------------------
    # normal_polynomial_automation('C:\\Users\\navea\\Desktop\\yarotsky_automation_normal')
    # --------------------------------------------------


    '''SECTION 8'''
    # oOoOoOoO  8A1  oOoOoOoO
    mse_before_training, mse_after_training = section_8A(N=3, taylor_degree=4, sampling_res=1e-7,
                                                                   learning_rate=1e-3, ud_relus=False, taylor_init = True,
                                                                  train_polynomial=True, train_bumps=True)
    print("Init MSE = {}\t\tFinal MSE = {}\n\n\n".format(mse_before_training, mse_after_training))

    # oOoOoOoO  8A2  oOoOoOoO
    mse_before_training, mse_after_training = section_8A(N=3, taylor_degree=4, sampling_res=1e-7,
                                                                  learning_rate=1e-3, ud_relus=False, taylor_init=False,
                                                                  train_polynomial=True, train_bumps=True)
    print("Init MSE = {}\t\tFinal MSE = {}\n\n\n".format(mse_before_training, mse_after_training))

    # oOoOoOoO  8B  oOoOoOoO
    mse_before_training, mse_after_training = section_8B(N=3, taylor_degree=4, sampling_res=1e-7,
                                                         learning_rate=1e-3, ud_relus=False)
    print("Init MSE = {}\t\tFinal MSE = {}\n\n\n".format(mse_before_training, mse_after_training))

    pass

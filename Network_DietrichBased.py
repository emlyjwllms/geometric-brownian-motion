import tensorflow as tf
from tensorflow.keras import layers

import keras
import keras.backend as K

import tensorflow_probability as tfp

import sys
import numpy as np

tfd = tfp.distributions

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
NUMBER_TYPE = tf.float64  # or tf.float32

STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have


class SDEIntegrators:
    """
    Implements the Euler-Maruyama integrator
    scheme used in integration of SDE.
    """

    def __init__(self):
        pass

    @staticmethod
    def euler_maruyama(xn, h, _f_sigma, rng):
        """
        Integration method for SDE, order 1/2 (strong) and 1 (weak) accurate.

        Parameters
        ----------
        xn
        h
        _f_sigma
        rng

        Returns
        -------

        """
        dW = rng.normal(loc=0, scale=np.sqrt(h), size=xn.shape)
        xk = xn.reshape(1, -1)  # we only allow a single point as input

        fk, sk = _f_sigma(xk)
        if np.prod(sk.shape) == xk.shape[-1]:
            skW = sk * dW
        else:
            sk = sk.reshape(xk.shape[-1], xk.shape[-1])
            skW = (sk @ dW.T).T
        return xk + h * fk + skW



class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    DIFF_TYPES = ["diagonal", "triangular", "spd"]

    @staticmethod
    def define_forward_model(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                             activation="tanh", dtype=tf.float64):

        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        network_x = inputs
        for i in range(n_layers):
            network_x = layers.Dense(n_dim_per_layer, activation=activation, dtype=dtype,
                                     name=name + "_hidden/dense_{}".format(i))(network_x)
        network_output = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(network_x)

        network = tf.keras.Model(inputs=inputs, outputs=network_output,
                                 name=name + "_forward_model")
        return network

    @staticmethod
    def define_gaussian_process(n_input_dimensions, n_output_dimensions, n_layers, n_dim_per_layer, name,
                                diffusivity_type="diagonal", activation="tanh", dtype=tf.float64):
        
        def make_tri_matrix(z):
            # first, make all eigenvalues positive by changing the diagonal to positive values
            z = tfp.math.fill_triangular(z)
            z2 = tf.linalg.diag(tf.linalg.diag_part(z))
            z = z - z2 + tf.abs(z2)  # this ensures the values on the diagonal are positive
            return z

        def make_spd_matrix(z):
            z = make_tri_matrix(z)
            return tf.linalg.matmul(z, tf.linalg.matrix_transpose(z))
        
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                name=name + "_mean_hidden_{}".format(i))(gp_x)
        gp_output_mean = layers.Dense(n_output_dimensions, dtype=dtype,
                                      name=name + "_output_mean", activation=None)(gp_x)

        # initialize with extremely small (not zero!) values so that it does not dominate the drift
        # estimation at the beginning of training
        small_init = 1e-2
        initializer = tf.keras.initializers.RandomUniform(minval=-small_init, maxval=small_init, seed=None)

        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                name=name + "_std_hidden_{}".format(i))(gp_x)
        if diffusivity_type=="diagonal":
            gp_output_std = layers.Dense(n_output_dimensions,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer,
                                         activation=lambda x: tf.nn.softplus(x) + STD_MIN_VALUE,
                                         name=name + "_output_std", dtype=dtype)(gp_x)
        elif diffusivity_type=="triangular":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the lower triangular matrix with positive eigenvalues on the diagonal.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_cholesky", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_tri_matrix)(gp_output_tril)
        elif diffusivity_type=="spd":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            # gp_output_std = layers.Lambda(lambda L: tf.linalg.matmul(L, tf.transpose(L)))(gp_output_tril)
        else:
            raise ValueError(f"Diffusivity type {diffusivity_type} not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        
        gp = tf.keras.Model(inputs,
                            [gp_output_mean, gp_output_std],
                            name=name + "_gaussian_process")
        return gp


class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    @staticmethod
    def __log(message, flush=True):
        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        LossAndErrorPrintingCallback.__log(
            "\rThe average loss for epoch {} is {:7.10f} ".format(
                epoch, logs["loss"]
            )
        )


class SDEIdentification:
    """
    Wrapper class that can be used for SDE identification.
    Needs a "tf.keras.Model" like the SDEApproximationNetwork or VAEModel to work.
    """

    def __init__(self, model):
        self.model = model

    def train_model(self, x_n, x_np1, validation_split=0.1, n_epochs=100, batch_size=1000, step_size=None,
                    callbacks=[]):
        print(f"training for {n_epochs} epochs with {int(x_n.shape[0] * (1 - validation_split))} data points"
              f", validating with {int(x_n.shape[0] * validation_split)}")

        if not (step_size is None):
            x_n = np.column_stack([step_size, x_n])
        y_full = np.column_stack([x_n, x_np1])

        if len(callbacks) == 0:
            callbacks.append(LossAndErrorPrintingCallback())

        hist = self.model.fit(x=y_full,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=validation_split,
                              callbacks=callbacks)
        return hist

    def drift_diffusivity(self, x):
        drift, std = self.model.call_xn(x)
        return K.eval(drift), K.eval(std)

    def sample_path(self, x0, step_size, NT, N_iterates, map_every_iteration=None):
        """
        Use the neural network to sample a path with the Euler Maruyama scheme.
        """
        step_size = tf.cast(np.array(step_size), dtype=NUMBER_TYPE)
        paths = [np.ones((N_iterates, 1)) @ np.array(x0).reshape(1, -1)]
        for it in range(NT):
            x_n = paths[-1]
            apx_mean, apx_scale = self.model.call_xn(x_n)
            x_np1 = tfd.MultivariateNormalDiag(
                loc=x_n + step_size * apx_mean,
                scale_diag=tf.math.sqrt(step_size) * apx_scale
            ).sample()

            x_i = keras.backend.eval(x_np1)
            if not (map_every_iteration is None):
                x_i = map_every_iteration(x_i)
            paths.append(x_i)
        return [
            np.row_stack([paths[k][i] for k in range(len(paths))])
            for i in range(N_iterates)
        ]


class SDEApproximationNetwork(tf.keras.Model):

    def __init__(self,
                 sde_model: tf.keras.Model,
                 step_size=None,
                 method="euler",
                 diffusivity_type="diagonal",
                 **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.method = method
        self.diffusivity_type = diffusivity_type

        SDEApproximationNetwork.verify(self.method)

    @staticmethod
    def verify(method):
        pass

    def get_config(self):
        return {
            "sde_model": self.sde_model,
            "step_size": self.step_size,
            "method": self.method,
            "diffusivity_type": self.diffusivity_type
        }

    @staticmethod
    def euler_maruyama_pdf(ynp1_, yn_, step_size_, model_, diffusivity_type="diagonal"):
        """
        This implies a very simple sde_model, essentially just a Gaussian process
        on x_n that predicts the drift and diffusivity.
        Returns log P(y(n+1) | y(n)) for the Euler-Maruyama scheme.

        Parameters
        ----------
        ynp1_ next point in time.
        yn_ current point in time.
        step_size_ step size in time.
        model_ sde_model that returns a (drift, diffusivity) tuple.
        diffusivity_type defines which type of diffusivity matrix will be used. See ModelBuilder.DIFF_TYPES.

        Returns
        -------
        logarithm of p(ynp1_ | yn_) under the Euler-Maruyama scheme.

        """
        drift_, diffusivity_ = model_(yn_)

        if diffusivity_type=="diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=(yn_ + step_size_ * drift_),
                scale_diag=tf.math.sqrt(step_size_) * diffusivity_,
                name="approx_normal"
            )
        elif diffusivity_type=="triangular":
            diffusivity_tril_ = diffusivity_

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # better use tfp.bijectors.FillScaleTriL()
            tril_step_size = tf.math.sqrt(step_size_)
            n_dim = K.shape(yn_)[-1]
            full_shape = n_dim * n_dim
            step_size_matrix = tf.broadcast_to(tril_step_size, [K.shape(step_size_)[0], full_shape])
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

            # now form the normal distribution
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=tf.multiply(step_size_matrix, diffusivity_tril_),
                name="approx_normal"
            )
        elif diffusivity_type=="spd":
            diffusivity_spd_ = diffusivity_

            # a cumbersome way to multiply the step size scalar with the batch of matrices...
            # TODO: REFACTOR with diffusivity_type=="triangular"
            spd_step_size = tf.math.sqrt(step_size_) # NO square root because we use cholesky below?
            n_dim = K.shape(yn_)[-1]
            full_shape = n_dim * n_dim
            step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_size_)[0], full_shape])
            step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

            # multiply with the step size
            covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
            # square the matrix so that the cholesky decomposition does not change the eienvalues
            covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
            # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
            covariance_matrix = tf.linalg.cholesky(covariance_matrix)
            
            # now form the normal distribution
            approx_normal = tfd.MultivariateNormalTriL(
                loc=(yn_ + step_size_ * drift_),
                scale_tril=covariance_matrix,
                name="approx_normal"
            )
        else:
            raise ValueError(f"Diffusivity type <{diffusivity_type}> not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        return approx_normal.log_prob(ynp1_)

    @staticmethod
    def split_inputs(inputs, step_size=None):
        if step_size is None:
            n_size = (inputs.shape[1] - 1) // 2
            step_size, x_n, x_np1 = tf.split(inputs, num_or_size_splits=[1, n_size, n_size], axis=1)
        else:
            step_size = step_size
            x_n, x_np1 = tf.split(inputs, num_or_size_splits=2, axis=1)
        return step_size, x_n, x_np1

    def call_xn(self, inputs_xn):
        """
        Can be used to evaluate the drift and diffusivity
        of the sde_model. This is different than the "call" method
        because it only expects "x_k", not "x_{k+1}" as well.
        """
        return self.sde_model(inputs_xn)


    def call(self, inputs):
        """
        Expects the input tensor to contain all of (step_sizes, x_k, x_{k+1}).
        """
        step_size, x_n, x_np1 = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(x_np1, x_n, step_size, self.sde_model,
                                                                  self.diffusivity_type)
        else:
            raise ValueError(self.method + " not available")

        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        loss = distortion

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return self.sde_model(x_n)
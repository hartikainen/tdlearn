import itertools
import logging

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tree

from td import OffPolicyValueFunctionPredictor


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


def cast_and_concat(x):
    x = tree.map_structure(
        lambda element: tf.cast(element, tf.float32), x)
    x = tree.flatten(x)
    x = tf.concat(x, axis=-1)
    return x


def feedforward_model(hidden_layer_sizes,
                      output_shape,
                      activation='relu',
                      output_activation='linear',
                      preprocessors=None,
                      name='feedforward_model',
                      *args,
                      **kwargs):
    output_size = tf.reduce_prod(output_shape)
    if 1 < len(output_shape):
        raise NotImplementedError("TODO(hartikainen)")
    model = tf.keras.Sequential([
        tfkl.Lambda(cast_and_concat)
    ] + [
        tf.keras.layers.Dense(
            hidden_layer_size, *args, activation=activation, **kwargs)
        for hidden_layer_size in hidden_layer_sizes
    ] + [
        tf.keras.layers.Dense(
            output_size, *args, activation=output_activation, **kwargs),
        tf.keras.layers.Reshape(output_shape),
    ], name=name)

    return model


class NonLinearBBO(OffPolicyValueFunctionPredictor):
    def __init__(self,
                 alpha,
                 D_s,
                 D_a,
                 prior_mean=0.0,
                 prior_stddev=0.1,
                 *args,
                 **kwargs):
        """TODO(hartikainen)."""
        self.D_s = D_s
        self.D_a = D_a
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        OffPolicyValueFunctionPredictor.__init__(self, *args, **kwargs)

        # self.uncertainty_model = bbo.OnlineUncertaintyModel()
        self.network = feedforward_model(
            hidden_layer_sizes=(32, 32),
            output_shape=(1, ),
            activation='relu',
            output_activation='linear',
        )
        self.V_fn = feedforward_model(
            hidden_layer_sizes=(32, 32),
            output_shape=(1, ),
            activation='relu',
            output_activation='linear',
        )

        self.network_optimizer = tf.optimizers.Adam(3e-4)
        self.V_fn_optimizer = tf.optimizers.Adam(3e-4)

        self.init_vals['alpha'] = alpha
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def clone(self):
        o = self.__class__(
            self.init_vals['alpha'], D_a=self.D_a, gamma=self.gamma, phi=self.phi)
        return o

    def __getstate__(self):
        res = self.__dict__.copy()
        for n in ("alpha", ):
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()

        network = res.pop('network')
        assert 'network' not in res, res['network']
        if network.built:
            res['network'] = {
                'config': network.get_config(),
                'weights': network.get_weights(),
            }
        else:
            res['network'] = {
                'config': network.get_config(),
                'weights': None,
            }

        V_fn = res.pop('V_fn')
        assert 'V_fn' not in res, res['V_fn']
        if V_fn.built:
            res['V_fn'] = {
                'config': V_fn.get_config(),
                'weights': V_fn.get_weights(),
            }
        else:
            res['V_fn'] = {
                'config': V_fn.get_config(),
                'weights': None,
            }

        return res

    def __setstate__(self, state):
        if hasattr(state, 'V_fn'):
            V_fn_state = state.pop('V_fn')
            V_fn_config = V_fn_state['config']
            V_fn_weights = V_fn_state['weights']
            V_fn = tf.keras.Sequential.from_config(V_fn_config)
            V_fn.set_weights(V_fn_weights)
        else:
            V_fn = feedforward_model(
                hidden_layer_sizes=(32, 32),
                output_shape=(1, ),
                activation='relu',
                output_activation='linear')

        state['V_fn'] = V_fn

        if hasattr(state, 'network'):
            network_state = state.pop('network')
            network_config = network_state['config']
            network_weights = network_state['weights']
            network = tf.keras.Sequential.from_config(network_config)
            network.set_weights(network_weights)
        else:
            network = feedforward_model(
                hidden_layer_sizes=(32, 32),
                output_shape=(1, ),
                activation='relu',
                output_activation='linear')

        state['network'] = network

        self.__dict__ = state.copy()
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    # def reset(self):
    #     LinearValueFunctionPredictor.reset(self)

    #     if self.uncertainty_model.built:
    #         self.uncertainty_model.reset()
    #     else:
    #         # self.uncertainty_model(np.zeros(self.phi.dim + self.D_a)[None, ...])
    #         if hasattr(self.phi, 'n'):
    #             self.uncertainty_model(np.zeros(self.phi.n)[None, ...])
    #         else:
    #             self.uncertainty_model(np.zeros(self.phi.dim)[None, ...])

    #     self.alpha = self._assert_iterator(self.init_vals['alpha'])
    #     self.w = np.zeros_like(self.init_vals['theta'])
    #     if hasattr(self, "A"):
    #         del(self.A)
    #     if hasattr(self, "b"):
    #         del(self.b)

    #     if hasattr(self, "F"):
    #         del(self.F)
    #     if hasattr(self, "Cmat"):
    #         del(self.Cmat)

    def init_deterministic(self, task):
        import ipdb; ipdb.set_trace(context=30)
        raise NotImplementedError("TODO(hartikainen)")

        self.F, self.Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(self.F - self.Cmat)

    def V(self, *args, **kwargs):
        return self.V_fn(*args, **kwargs).numpy()

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        if theta is None:
            theta = self.theta

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        b_N = np.concatenate((f0, ))[None, ...].astype(np.float32)
        b_hat = np.concatenate((f0, ))[None, ...].astype(np.float32)
        b_N_next = np.concatenate((f1, ))[None, ...].astype(np.float32)

        tau = r + self.gamma * self.network(b_N_next)

        with tf.GradientTape(persistent=True) as tape:
            f = self.network(b_N)

        f_grads = tape.gradient(f, self.network.trainable_variables)

        phi_0s = tree.map_structure(
            lambda phi: tf.random.normal(
                tf.shape(phi), mean=self.prior_mean, stddev=self.prior_stddev),
            self.network.trainable_variables)

        phi_diffs = tree.map_structure(
            lambda phi, phi_0: phi - phi_0,
            self.network.trainable_variables, phi_0s)

        assert (f - tau).shape == [1, 1]
        phi_updates = tree.map_structure(
            lambda f_grad, phi_diff: (f - tau)[0] * f_grad + phi_diff,
            f_grads, phi_diffs)

        self.network_optimizer.apply_gradients(
            zip(phi_updates, self.network.trainable_variables))

        # f_gradients = (f - target) * grad_f + phi_diffs

        b_j = [b_N, b_N_next][np.random.choice(2)]

        with tf.GradientTape() as tape:
            V_j = self.V_fn(b_j)

        V_grads = tape.gradient(V_j, self.V_fn.trainable_variables)
        f_j = self.network(b_j)

        assert (V_j - f_j).shape == [1, 1]
        omega_updates = tree.map_structure(
            lambda V_grad: (V_j - f_j)[0] * V_grad, V_grads)

        self.V_fn_optimizer.apply_gradients(
            zip(omega_updates, self.V_fn.trainable_variables))

        self._toc()

        return self.theta


class NonLinearTD0(OffPolicyValueFunctionPredictor):

    """
    TD(0) learning algorithm for on- and off-policy value function estimation
    with linear function approximation
    for details on off-policy importance weighting formulation see
    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    University of Alberta. (p. 31)
    """

    def __init__(self, alpha, D_s, D_a, **kwargs):
        """
            alpha:  step size. This can either be a constant number or
                    an iterable object providing step sizes
            gamma:  discount factor
        """
        self.D_s = D_s
        self.D_a = D_a
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        self.init_vals['alpha'] = alpha
        self.reset()

    def clone(self):
        o = self.__class__(
            alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        self.V_fn = feedforward_model(
            hidden_layer_sizes=(32, 32),
            output_shape=(1, ),
            activation='relu',
            output_activation='linear',
        )

        self.V_fn(np.zeros([1, self.D_s]))

        self.optimizer = tf.optimizers.Adam(3e-4)
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def __getstate__(self):
        res = self.__dict__
        for n in ["alpha"]:
            if isinstance(res[n], itertools.repeat):
                res[n] = res[n].next()

        V_fn = res.pop('V_fn')
        assert 'V_fn' not in res, res['V_fn']
        if V_fn.built:
            res['V_fn'] = {
                'config': V_fn.get_config(),
                'weights': V_fn.get_weights(),
            }
        else:
            res['V_fn'] = {
                'config': V_fn.get_config(),
                'weights': None,
            }

        return res

    def __setstate__(self, state):
        if hasattr(state, 'V_fn'):
            V_fn_state = state.pop('V_fn')
            V_fn_config = V_fn_state['config']
            V_fn_weights = V_fn_state['weights']
            V_fn = tf.keras.Sequential.from_config(V_fn_config)
            V_fn.set_weights(V_fn_weights)
        else:
            V_fn = feedforward_model(
                hidden_layer_sizes=(32, 32),
                output_shape=(1, ),
                activation='relu',
                output_activation='linear')

        state['V_fn'] = V_fn

        self.__dict__ = state
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def V(self, *args, **kwargs):
        return self.V_fn(*args, **kwargs).numpy()

    @property
    def theta(self):
        return tf.concat([
            tf.reshape(x, [-1])
            for x in self.V_fn.trainable_variables
        ], axis=0)

    # @tf.function(experimental_relax_shapes=True)
    def update_V(self, s0, s1, r, f0=None, f1=None, theta=None, rho=1, **kwargs):
        """
        adapt the current parameters theta given the current transition
        (s0 -> s1) with reward r and (a weight of rho)
        returns the next theta
        """
        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        self._tic()

        target = r + self.gamma * self.V_fn(f1[None])[0]
        with tf.GradientTape() as tape:
            value = self.V_fn(f0[None])[0]
            loss = 0.5 * tf.losses.MSE(y_true=target, y_pred=value)

        gradients = tape.gradient(loss, self.V_fn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.V_fn.trainable_variables))

        logging.debug("TD Learning Delta {}".format(loss))
        print("TD Learning Delta {}".format(loss))

        # al = self.alpha.next()
        # # if isinstance(self.alpha,  RMalpha):
        # #    print al, self.alpha.t
        # theta += al * delta * rho * f0
        # self.theta = theta
        self._toc()
        return 0.0

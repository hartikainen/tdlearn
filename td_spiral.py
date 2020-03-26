import itertools
import logging

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tree

from td import OffPolicyValueFunctionPredictor


class SpiralModel(tf.keras.Model):
    def __init__(self, initial_omega, epsilon=1e-6):
        super(SpiralModel, self).__init__()
        self._initial_omega = initial_omega
        self._epsilon = epsilon
        self.omega = self.add_weight(
            'mu_hat',
            shape=(),
            initializer=tf.initializers.constant(initial_omega))

    def call(self, inputs):
        tf.debugging.assert_equal(tf.rank(inputs), 2)
        tf.debugging.assert_equal(tf.shape(inputs)[1], 3)
        tf.debugging.assert_equal(
            tf.reduce_sum(tf.cast(inputs == 0, tf.float32), axis=1), 2.0)
        tf.debugging.assert_equal(
            tf.reduce_sum(tf.cast(inputs == 1, tf.float32), axis=1), 1.0)

        x = (tf.sqrt(3.0) * self.omega) / 2.0
        V = tf.exp(self._epsilon * self.omega) * tf.stack((
            tf.sqrt(3.0) * tf.sin(x) - tf.cos(x),
            - tf.sqrt(3.0) * tf.sin(x) - tf.cos(x),
            2.0 * tf.cos(x),
        ))

        result = tf.linalg.matvec(inputs, V)[..., None]
        tf.debugging.assert_equal(tf.rank(result), 2)
        tf.debugging.assert_equal(tf.shape(result)[0], tf.shape(inputs)[0])
        tf.debugging.assert_equal(tf.shape(result)[1], 1)

        return result

    def get_config(self):
        config = {
            'initial_omega': self._initial_omega,
            'epsilon': self._epsilon,
        }
        return config


class SpiralNonLinearBBO(OffPolicyValueFunctionPredictor):
    def __init__(self,
                 alpha,
                 D_s,
                 D_a,
                 initial_omega,
                 epsilon,
                 prior_mean=0.0,
                 prior_stddev=0.1,
                 V_fn_lr=3e-4,
                 network_lr=3e-4,
                 *args,
                 **kwargs):
        """TODO(hartikainen)."""
        self.D_s = D_s
        self.D_a = D_a
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        self._initial_omega = initial_omega
        self._epsilon = epsilon

        self._network_lr = network_lr
        self._V_fn_lr = V_fn_lr

        OffPolicyValueFunctionPredictor.__init__(self, *args, **kwargs)

        self.init_vals['alpha'] = alpha
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

        self.reset()

    def clone(self):
        o = self.__class__(
            self.init_vals['alpha'], D_a=self.D_a, gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        OffPolicyValueFunctionPredictor.reset(self)

        if not hasattr(self, 'network'):
            self.network = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)
            self.V_fn = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)
            self.network_optimizer = tf.optimizers.SGD(self._network_lr)
            self.V_fn_optimizer = tf.optimizers.SGD(self._V_fn_lr)

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
        V_fn_state = state.pop('V_fn', None)
        network_state = state.pop('network', None)

        self.__dict__ = state.copy()

        if V_fn_state is not None:
            V_fn_config = V_fn_state['config']
            V_fn_weights = V_fn_state['weights']
            V_fn = SpiralModel(**V_fn_config)
            V_fn.set_weights(V_fn_weights)
        else:
            V_fn = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)

        self.V_fn = V_fn

        if network_state is not None:
            network_config = network_state['config']
            network_weights = network_state['weights']
            network = SpiralModel(**network_config)
            network.set_weights(network_weights)
        else:
            network = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)

        self.network = network
        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def init_deterministic(self, task):
        import ipdb; ipdb.set_trace(context=30)
        raise NotImplementedError("TODO(hartikainen)")

        self.F, self.Cmat, self.b = self._compute_detTD_updates(task)
        self.A = np.array(self.F - self.Cmat)

    def V(self, *args, **kwargs):
        return self.V_fn(*args, **kwargs).numpy().flatten()

    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        if theta is None:
            theta = self.theta

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        assert f0.size == 3 and f1.size == 3

        self._tic()

        b_N = np.atleast_2d(f0)
        b_hat = np.atleast_2d(f0)
        b_N_next = np.atleast_2d(f1)

        target = r + self.gamma * self.V_fn(b_N_next)

        # assert np.rank(target) == 1, (target, target.shape)
        assert np.size(target) == 1, (target, target.shape)

        with tf.GradientTape() as tape:
            f = self.network(b_N)

            prior_loss = tf.reduce_sum(tree.map_structure(
                lambda phi: tf.reduce_mean(
                    tf.losses.MSE(phi[None], tf.random.normal(
                        tf.shape(phi[None]),
                        mean=self.prior_mean,
                        stddev=self.prior_stddev))),
                tree.flatten(self.network.trainable_variables)))

            f_loss = 1.0 * (
                tf.losses.MSE(y_true=tf.stop_gradient(target), y_pred=f)
                + prior_loss)

        # assert prior_loss == 0.0, prior_loss

        f_gradients = tape.gradient(f_loss, self.network.trainable_variables)
        self.network_optimizer.apply_gradients(
            zip(f_gradients, self.network.trainable_variables))

        b_j = np.atleast_2d(self.phi(np.array([np.random.choice(self.D_s)])))
        # b_j = [b_N, b_N_next][np.random.choice(2)]
        assert b_j.shape == b_N.shape

        f_j = self.network(b_j)
        with tf.GradientTape() as tape:
            V_j = self.V_fn(b_j)
            V_loss = 1.0 * tf.losses.MSE(
                y_true=tf.stop_gradient(f_j), y_pred=V_j)

        V_gradients = tape.gradient(V_loss, self.V_fn.trainable_variables)
        self.V_fn_optimizer.apply_gradients(
            zip(V_gradients, self.V_fn.trainable_variables))

        print("V_loss: {:.3f}, f_loss: {:.3f}, f_i: {:.3f}, tau: {:.3f}, f_j: {:.3f}, V_j: {:.3f}"
              "".format(
                  V_loss.numpy().squeeze().item(),
                  f_loss.numpy().squeeze().item(),
                  f.numpy().squeeze().item(),
                  target.numpy().squeeze().item(),
                  f_j.numpy().squeeze().item(),
                  V_j.numpy().squeeze().item()))

        self._toc()

        return self.theta


class SpiralNonLinearBilevel(SpiralNonLinearBBO):
    def update_V(self, s0, s1, r, f0=None, f1=None, rho=1, theta=None, **kwargs):
        assert 0 <= s0 and s0 < 3, s0
        assert 0 <= s1 and s1 < 3, s1
        assert s1 != ((s0 + 1) % 3)

        if f0 is None or f1 is None:
            f0 = self.phi(s0)
            f1 = self.phi(s1)

        assert f0.size == 3 and f1.size == 3

        self._tic()

        b_N = np.atleast_2d(f0)
        b_N_next = np.atleast_2d(f1)

        theta_0 = self.network.get_weights()[0].copy()
        omega_0 = self.V_fn.get_weights()[0].copy()

        target = r + self.gamma * self.V_fn(b_N_next)
        assert np.size(target) == 1, (target, target.shape)
        target = tf.squeeze(target)

        with tf.GradientTape() as tape:
            f = self.network(b_N)
            assert np.size(f) == 1, (f, f.shape)
            f = tf.squeeze(f)

        grad_f = tape.gradient(f, self.network.trainable_variables[0])
        beta = self._network_lr
        delta = f - target
        theta_1 = theta_0 - beta * delta * grad_f
        self.network.set_weights([theta_1])

        alpha = self._V_fn_lr
        omega_1 = (1 - alpha) * omega_0 + alpha * theta_0
        self.V_fn.set_weights([omega_1])

        _V_loss = np.abs(omega_0 - theta_0)
        print("V_loss: {:.3f}, f_loss: {:.3f}, f_i: {:.3f}, tau: {:.3f}, s0: {}, s1: {}"
              "".format(
                  _V_loss.item(),
                  delta.numpy().squeeze().item(),
                  f.numpy().squeeze().item(),
                  target.numpy().squeeze().item(),
                  int(s0),
                  int(s1),
              ))

        self._toc()

        return self.theta


class SpiralNonLinearTD0(OffPolicyValueFunctionPredictor):
    """
    TD(0) learning algorithm for on- and off-policy value function estimation
    with linear function approximation
    for details on off-policy importance weighting formulation see
    Maei, H. R. (2011). Gradient Temporal-Difference Learning Algorithms.
    University of Alberta. (p. 31)
    """

    def __init__(self,
                 alpha,
                 D_s,
                 D_a,
                 initial_omega,
                 epsilon,
                 **kwargs):
        self.D_s = D_s
        self.D_a = D_a
        self._initial_omega = initial_omega
        self._epsilon = epsilon
        self.V_fn_lr = alpha
        OffPolicyValueFunctionPredictor.__init__(self, **kwargs)
        self.init_vals['alpha'] = alpha
        self.reset()

    def clone(self):
        o = self.__class__(
            alpha=self.init_vals['alpha'], gamma=self.gamma, phi=self.phi)
        return o

    def reset(self):
        OffPolicyValueFunctionPredictor.reset(self)

        if not hasattr(self, 'network'):
            self.V_fn = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)
            self.optimizer = tf.optimizers.SGD(self.V_fn_lr)
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
        V_fn_state = state.pop('V_fn', None)
        self.__dict__ = state

        if V_fn_state is not None:
            V_fn_config = V_fn_state['config']
            V_fn_weights = V_fn_state['weights']
            V_fn = SpiralModel(**V_fn_config)
            V_fn.set_weights(V_fn_weights)
        else:
            V_fn = SpiralModel(
                initial_omega=self._initial_omega,
                epsilon=self._epsilon)

        self.V_fn = V_fn

        self.alpha = self._assert_iterator(self.init_vals['alpha'])

    def V(self, *args, **kwargs):
        return self.V_fn(*args, **kwargs).numpy().flatten()

    # @property
    # def theta(self):
    #     import ipdb; ipdb.set_trace(context=30)
    #     return tf.concat([
    #         tf.reshape(x, [-1])
    #         for x in self.V_fn.trainable_variables
    #     ], axis=0)

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

        f0 = np.atleast_2d(f0)
        f1 = np.atleast_2d(f1)
        assert f0.size == 3 and f1.size == 3
        target = r + self.gamma * self.V_fn(f1)[0]
        assert np.rank(target) == 1, (target, target.shape)
        assert np.size(target) == 1, (target, target.shape)
        with tf.GradientTape() as tape:
            value = self.V_fn(f0)[0]
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

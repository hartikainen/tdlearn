from __future__ import division

from collections import OrderedDict
import tensorflow as tf
import tree


@tf.function(experimental_relax_shapes=True)
def batch_quadratic_form(W, inputs):
    W = tf.cast(tf.convert_to_tensor(W), tf.float32)
    inputs = tf.cast(tf.convert_to_tensor(inputs), tf.float32)
    return tf.einsum(
        '...bi,ji,...bj->...b', inputs, W, inputs
    )[..., tf.newaxis]


class OnlineUncertaintyModel(tf.keras.Model):
    def build(self, input_shapes):
        D = sum(input_shape[-1] for input_shape in tree.flatten(input_shapes))
        self.mu_hat = self.add_weight(
            'mu_hat', shape=(1, D), initializer=tf.initializers.zeros)
        self.Sigma_hat = self.add_weight(
            'Sigma_hat', shape=(D, D), initializer=tf.initializers.identity)
        self.Sigma_N = self.add_weight(
            'Sigma_N', shape=(D, D), initializer=tf.initializers.identity)
        self.Delta_N = self.add_weight(
            'Delta_N', shape=(D, D), initializer=tf.initializers.zeros)
        self.N = self.add_weight(
            'N', shape=(), dtype=tf.int32, initializer=tf.initializers.zeros)

    def reset(self):
        self.mu_hat.assign(tf.zeros_like(self.mu_hat))
        self.Sigma_hat.assign(tf.eye(*self.Sigma_hat.shape))
        self.Sigma_N.assign(tf.eye(*self.Sigma_N.shape))
        self.Delta_N.assign(tf.zeros_like(self.Delta_N))
        self.N.assign(tf.zeros_like(self.N))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        mu_hat = self.mu_hat
        Sigma_N = self.Sigma_N
        Sigma_hat = self.Sigma_hat

        mu_hat_mu_hat_T = tf.matmul(
            mu_hat[..., None], mu_hat[..., None], transpose_b=True
        )[0]
        epistemic_uncertainty = tf.linalg.trace(
            tf.matmul(Sigma_hat - mu_hat_mu_hat_T, Sigma_N)
        ) + tf.squeeze(batch_quadratic_form(Sigma_N, mu_hat))

        return epistemic_uncertainty

    @tf.function(experimental_relax_shapes=True)
    def online_update(self, b_N, b_hat, b_N_next, gamma):
        N = tf.shape(b_N)[0]
        tf.debugging.assert_equal(N, 1)

        variance = 1.0
        b_b_T = tf.matmul(b_N, b_N, transpose_a=True)

        def initialize_Sigma_N():
            Sigma_N_inv = tf.linalg.inv(self.Sigma_N) + b_b_T / variance
            cholesky = tf.linalg.cholesky(tf.cast(Sigma_N_inv, tf.float64))
            Sigma_N = tf.linalg.cholesky_solve(
                cholesky,
                tf.eye(tf.shape(self.Sigma_N)[-1], dtype=tf.float64))
            return tf.cast(Sigma_N, tf.float32)

        def update_Sigma_N():
            Sigma_N_delta = - (
                tf.matmul(tf.matmul(self.Sigma_N, b_b_T), self.Sigma_N)
                / (variance + tf.reduce_mean(batch_quadratic_form(self.Sigma_N, b_N))))
            Sigma_N = self.Sigma_N + Sigma_N_delta
            return Sigma_N

        Sigma_N = tf.cond(
            tf.equal(self.N, 0),
            initialize_Sigma_N,
            update_Sigma_N)

        self.Sigma_N.assign(Sigma_N)

        Delta_N_delta = tf.matmul(
            (gamma * b_N_next - b_N), b_N, transpose_a=True)
        self.Delta_N.assign_add(Delta_N_delta)

        mu_hat = (
            tf.cast(self.N / (self.N + N), b_hat.dtype)
            * self.mu_hat
        ) + (
            tf.cast(N / (self.N + N), b_hat.dtype)
            * tf.reduce_mean(b_hat, axis=0))

        self.mu_hat.assign(mu_hat)

        b_hat_b_hat_T = tf.matmul(
            b_hat[..., None], b_hat[..., None], transpose_b=True)

        Sigma_hat = (
            tf.cast(self.N / (self.N + N), b_hat_b_hat_T.dtype)
            * self.Sigma_hat
        ) + (
            tf.cast(N / (self.N + N), b_hat_b_hat_T.dtype)
            * tf.reduce_mean(b_hat_b_hat_T, axis=0))

        self.Sigma_hat.assign(Sigma_hat)

        self.N.assign_add(N)

        return True

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', self.N.numpy()),
            ('epistemic_uncertainty', self(True).numpy()),
        ))
        return diagnostics

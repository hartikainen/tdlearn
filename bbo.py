

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
    def build(self, input_shapes, sigma_0=1.0):
        self.sigma_0 = sigma_0
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

        variance = self.sigma_0
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


class OnlineUncertaintyModelV2(tf.keras.Model):
    def __init__(self, sigma_0=1.0):
        super(OnlineUncertaintyModelV2, self).__init__()
        self.sigma_0 = sigma_0

    def build(self, input_shapes):
        D = sum(input_shape[-1] for input_shape in tree.flatten(input_shapes))
        self.C_inverse = self.add_weight(
            'C_inverse',
            shape=(D, D),
            initializer=tf.initializers.identity(gain=-1.0 * self.sigma_0))
        self.rho = self.add_weight(
            'rho', shape=(1, D), initializer=tf.initializers.zeros)
        self.Delta_N = self.add_weight(
            'Delta_N', shape=(D, D), initializer=tf.initializers.zeros)
        self.N = self.add_weight(
            'N', shape=(), dtype=tf.int32, initializer=tf.initializers.zeros)

    def reset(self):
        self.C_inverse.assign(-1.0 * self.sigma_0 * tf.eye(*self.C_inverse.shape))
        self.rho.assign(tf.zeros_like(self.rho))
        self.Delta_N.assign(tf.zeros_like(self.Delta_N))
        self.N.assign(tf.zeros_like(self.N))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        return 0.0

    @tf.function(experimental_relax_shapes=True)
    def online_update(self, b_N, b_hat, b_N_next, gamma, reward, iw=1.0):
        N = tf.shape(b_N)[0]
        tf.debugging.assert_equal(N, 1)

        Delta = iw * (gamma * b_N_next - b_N)

        def update_C_inverse():
            Delta_C_inverse = tf.matmul(Delta, self.C_inverse)
            C_inverse_delta = - 1.0 * tf.matmul(
                tf.matmul(self.C_inverse, b_N, transpose_b=True),
                Delta_C_inverse
            ) / (1.0 + tf.matmul(Delta_C_inverse, b_N, transpose_b=True))

            C_inverse = self.C_inverse + C_inverse_delta
            return C_inverse

        C_inverse = update_C_inverse()
        self.C_inverse.assign(C_inverse)

        rho_delta = iw * b_N * reward
        self.rho.assign_add(rho_delta)

        self.N.assign_add(N)

        return True

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', self.N.numpy()),
            ('epistemic_uncertainty', self(True).numpy()),
        ))
        return diagnostics


class OnlineUncertaintyModelV3(tf.keras.Model):
    def __init__(self, sigma_0=1.0):
        super(OnlineUncertaintyModelV3, self).__init__()
        self.sigma_0 = sigma_0

    def build(self, input_shapes):
        D = sum(input_shape[-1] for input_shape in tree.flatten(input_shapes))
        self.C_inverse = self.add_weight(
            'C_inverse',
            shape=(D, D),
            initializer=tf.initializers.identity(gain=-1.0 * self.sigma_0))
        self.rho = self.add_weight(
            'rho', shape=(1, D), initializer=tf.initializers.zeros)
        self.N = self.add_weight(
            'N', shape=(), dtype=tf.int32, initializer=tf.initializers.zeros)

    def reset(self):
        self.C_inverse.assign(tf.eye(*self.C_inverse.shape) * -1.0 * self.sigma_0)
        self.rho.assign(tf.zeros_like(self.rho))
        self.N.assign(tf.zeros_like(self.N))

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs):
        return 0.0

    @tf.function(experimental_relax_shapes=True)
    def online_update(self, b_N, b_hat, b_N_next, gamma, reward, iw=1.0):
        N = tf.shape(b_N)[0]
        tf.debugging.assert_equal(N, 1)

        Delta = iw * (gamma * b_N_next - b_N)

        def update_C_inverse():
            Delta_C_inverse = tf.matmul(Delta, self.C_inverse)
            C_inverse_delta = - 1.0 * tf.matmul(
                tf.matmul(self.C_inverse, b_N, transpose_b=True),
                Delta_C_inverse
            ) / (1.0 + tf.matmul(Delta_C_inverse, b_N, transpose_b=True))

            C_inverse = self.C_inverse + C_inverse_delta
            return C_inverse

        C_inverse = update_C_inverse()
        C_inverse_assign = self.C_inverse.assign(C_inverse)

        def update_C_inverse_2():
            # tf.print("self.N: ", self.N)
            C_inverse_nth_col = self.C_inverse[:, self.N]
            C_inverse_nth_row = self.C_inverse[self.N, :]

            # C_inverse_nth_column_row_outer_product = tf.einsum(
            #     'i,j->ij', C_inverse_nth_row, C_inverse_nth_col)
            C_inverse_nth_column_row_outer_product = tf.einsum(
                'i,j->ij', C_inverse_nth_col, C_inverse_nth_row)
            C_inverse_nth_diagonal = self.C_inverse[self.N, self.N]
            C_inverse_delta = - 1.0 * (
                (self.sigma_0 * C_inverse_nth_column_row_outer_product)
                / (1.0 + self.sigma_0 * C_inverse_nth_diagonal))

            C_inverse = self.C_inverse + C_inverse_delta
            return C_inverse

        rho_delta = iw * b_N * reward
        self.rho.assign_add(rho_delta)

        with tf.control_dependencies([C_inverse_assign]):
            C_inverse = tf.cond(
                tf.less(self.N, tf.shape(self.C_inverse)[0]),
                update_C_inverse_2,
                lambda: self.C_inverse)

            C_inverse_assign = self.C_inverse.assign(C_inverse)

            with tf.control_dependencies([C_inverse_assign]):
                self.N.assign_add(N)

        tf.debugging.check_numerics(self.C_inverse, "C_inverse")
        tf.debugging.check_numerics(self.rho, "rho")

        return True

    def get_diagnostics(self):
        diagnostics = OrderedDict((
            ('N', self.N.numpy()),
            ('epistemic_uncertainty', self(True).numpy()),
        ))
        return diagnostics

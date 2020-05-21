import tensorflow as tf


def set_gpu_memory_growth(growth):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, growth)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print((len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs"))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def split_gpus(num_logical_devices_per_gpu):
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.set_logical_device_configuration(
            physical_device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=1000)
             for _ in range(num_logical_devices_per_gpu)])

import numpy as np
import tensorflow as tf
import multiprocessing
import random

def guess_available_cpus():
    return int(multiprocessing.cpu_count())

def setup_tensorflow_session():
    # i do not want too much overhead on my cpus
    # because i will be running multiple experiments at the same time
    num_cpu = guess_available_cpus()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.69,
                                allow_growth=True)

    # check if using num_cpu = 32 (our max) causes performance problems???
    tf_config = tf.ConfigProto(gpu_options=gpu_options,
                               inter_op_parallelism_threads=num_cpu,
                               intra_op_parallelism_threads=num_cpu,
                               allow_soft_placement=True)
    return tf.Session(config=tf_config)

def set_global_seeds(seed):
    import tensorflow as tf
    from gym.utils.seeding import hash_seed
    seed = hash_seed(seed, max_bytes=4)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

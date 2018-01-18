
import tensorflow as tf


def get_input_fn_dataset(dataset_name = 'train', num_epoch=30, batch_size=256):
    def _parse_function(example_proto):
        features = {"X": tf.VarLenFeature(tf.float32),
                  "Y": tf.VarLenFeature(tf.float32)}
        features = {'id': tf.FixedLenFeature([1], tf.int64),
                    'X': tf.FixedLenFeature([300, 100], tf.float32),
                    'Y': tf.FixedLenFeature([6], tf.float32)}
        parsed_features = tf.parse_single_example(example_proto, features)
        #return parsed_features["X"], parsed_features["Y"]
        return parsed_features["id"], parsed_features['X'], parsed_features['Y']

    def input_fn():
        dataset = tf.data.TFRecordDataset('data/{}.tfrecord'.format(dataset_name), compression_type='')
        #dataset = dataset.repeat(num_epoch)
        #dataset = dataset.shuffle(10*batch_size)
        #dataset = dataset.batch(batch_size)
        #dataset = tf.contrib.data.map_and_batch()
        dataset = dataset.map(_parse_function)
        #dataset = dataset.prefetch(10)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    """
    a model_fn for Estimator class
    This function will be called to create a new graph each time an estimator method is called
    """
    tf.keras.backend.set_learning_phase(mode == tf.estimator.ModeKeys.TRAIN)
    learning_rate = params['learning_rate']
    nb_class = 10

    X = features
    logits = 0  # <-- TODO: definir le model avec lstm ici

    predictions = {'class': tf.argmax(logits, axis=1), 'image': X}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    return tf.estimator.EstimatorSpec(predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      mode=mode,
                                      training_hooks=[log_hook],
                                      evaluation_hooks=[],
                                      eval_metric_ops={'acc_validation': accuracy_metric})

def main():
    input_fn = get_input_fn_dataset()
    import IPython
    IPython.embed()

    config= tf.estimator.RunConfig(save_summary_steps=10,
                                   save_checkpoints_steps=1000,
                                   keep_checkpoint_max=200,
                                   log_step_count_steps=1000)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir='mnist_trained',
                                       params={'learning_rate': 0.01},
                                       config=config)




if __name__ == "__main__":
    main()
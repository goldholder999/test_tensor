import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

class OzRNNManager:

    def __init__(self):

        # Prepare the data for training
        self.BUFFER_SIZE = 10000
        self.BATCH_SIZE = 64

        self.history = None
        self.model = None
        self.encoder = None
        self.train_dataset = None
        self.test_dataset = None

    def make_dataset(self):
        dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                                  as_supervised=True)
        self.train_dataset, self.test_dataset = dataset['train'], dataset['test']
        self.encoder = info.features['text'].encoder

        print('Vocabulary size: {}'.format(self.encoder.vocab_size))

        sample_string = 'Hello TensorFlow.'

        encoded_string = self.encoder.encode(sample_string)
        print('Encoded string is {}'.format(encoded_string))

        original_string = self.encoder.decode(encoded_string)
        print('The original string: "{}"'.format(original_string))

        assert original_string == sample_string

        for index in encoded_string:
            print('{} ----> {}'.format(index, self.encoder.decode([index])))

        self.train_dataset = self.train_dataset.shuffle(self.BUFFER_SIZE)
        self.train_dataset = self.train_dataset.padded_batch(self.BATCH_SIZE, self.train_dataset.output_shapes)

        self.test_dataset = self.test_dataset.padded_batch(self.BATCH_SIZE, self.test_dataset.output_shapes)

    def loadModel(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def make_model(self, model_type=1):
        if model_type == 1:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.encoder.vocab_size, 64),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.encoder.vocab_size, 64),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

    def fit(self):
        if self.train_dataset is None or self.test_dataset is None or self.model is None:
            print('data set not founnd')
            return
        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])
        self.history = self.model.fit(self.train_dataset, epochs=10,
                                      validation_data=self.test_dataset,
                                      validation_steps=30)
        # 저장
        self.model.save('rnnModel.hdf5')

    def evaluate(self):
        if self.model is None:
            return
        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))

    @classmethod
    def __plot_graphs(cls, history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string], '')
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    def plot_graphs(self, string):
        self.__plot_graphs(self.history, string)

    @classmethod
    def pad_to_size(cls, vec, size):
        zeros = [0] * (size - len(vec))
        vec.extend(zeros)
        return vec

    def sample_predict(self, sentence, pad):
        if self.encoder is None or self.model is None:
            return None
        encoded_sample_pred_text = self.encoder.encode(sentence)

        if pad:
            encoded_sample_pred_text = self.pad_to_size(encoded_sample_pred_text, 64)
        encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
        predictions = self.model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

        return (predictions)

    def print_model_summary(self):
        if self.model:
            self.model.summary()

import tensorflow as tf
import numpy as np
import pandas as pd
import os

from tensorflow.python.lib.io import file_io
from sklearn.metrics import roc_auc_score as auc
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

class simple_classifier:
    ''' A simple feed-forward classifier in TensorFlow'''
    def __init__(self):
        self.num_epochs = 50
        self.batch_size = 100
        self.display = 1
        self.test_ratio = 0.25
        self.x = tf.placeholder("float", [None, 18], name='features')
        self.y = tf.placeholder("float", shape=(None,2), name='target')

    def preprocess_data(self):
        with file_io.FileIO('/Users/patricksmith/desktop/creditcard.csv', mode ='r') as f:
            fraud_data = pd.read_csv(f)
        fraud_data['normAmount'] = StandardScaler().fit_transform(fraud_data['Amount'].values.reshape(-1, 1))
        fraud_data = fraud_data.drop(['Time','Amount','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
        x_data = pd.concat([fraud_data.iloc[:,0:17], fraud_data.iloc[:,18]], axis=1)
        y_data = fraud_data.iloc[:,17]
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=self.test_ratio, random_state=42)
        return x_train, x_test, y_train, y_test

    def mlp(self, x):
        initializer = tf.contrib.layers.xavier_initializer()
        h0 =  tf.layers.dense(x, x.shape[1], tf.nn.relu, kernel_initializer=initializer)
        h1 = tf.layers.dense(h0, 2, activation=None)
        logits = tf.nn.softmax(h1)
        return logits

    def train_model(self, *args, **kwargs):

        ## Parse the Arguments
        saved_args = locals()
        args_dict = saved_args['kwargs']
        job_dir_type = args_dict['job_dir']

        ## Gather the data
        x_train, x_test, y_train, y_test = self.preprocess_data()

        ## The model requires needs a column for each class as input
        y_train = np.array([y_train, -(y_train-1)]).T

        ## Prepare the Network
        logits = self.mlp(self.x)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))
        training_operation = tf.train.AdamOptimizer(2e-4).minimize(loss)

        ## Initialize the variables
        init = tf.global_variables_initializer()
        total_batch = int(len(x_train)/self.batch_size)

        ## Create the batches
        X_batches = np.array_split(x_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)

        ## Initialize the model saver
        save_model = os.path.join(job_dir, 'saved_classifier.ckpt')
        saver = tf.train.Saver()

        ## Start the training session
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.num_epochs):
                for i in range(total_batch):
                    batch_x, batch_y = X_batches[i], Y_batches[i]
                    _, c = sess.run([training_operation, loss], feed_dict={self.x: batch_x, self.y: batch_y})
                if epoch % self.display == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                    "cost=", "{:.9f}".format(c))

            ## Save the model checkpoints
            #saver.save(sess, save_model)
            predicted_indicies = tf.argmax(logits, 1)

            ## Model Serving Inputs
            inputs={"x": self.x}

            ## Model Serving Outputs
            outputs={
                "probs": logits,
                "pred_indicies": predicted_indicies
                }
            saved = os.path.join(job_dir, 'binaries')

            tf.saved_model.simple_save(sess, saved, inputs, outputs)
            print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models'
     )

     args, unknown = parser.parse_known_args()
     c = SimpleClassifier()
     c.train_model(**args.__dict__)

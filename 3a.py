# TODO popuniti kodom za problem 3a
#setup
%tensorflow_version 1.x
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

#algoritam
class KNN:
  def __init__(self, nb_features, nb_classes, data, k):
    self.nb_features = nb_features
    self.nb_classes = nb_classes
    self.data = data
    self.k = k

    self.X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Q = tf.placeholder(shape=(nb_features), dtype=tf.float32)


    dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), 
                                  axis=1))
    _, idxs = tf.nn.top_k(-dists, self.k)  


    self.classes = tf.gather(self.Y, idxs)
    self.dists = tf.gather(dists, idxs)

    self.w = tf.fill([k], 1/k)

    w_col = tf.reshape(self.w, (k, 1))
    self.classes_one_hot = tf.one_hot(self.classes, nb_classes)
    self.scores = tf.reduce_sum(w_col * self.classes_one_hot, axis=0)

    self.hyp = tf.argmax(self.scores)

  def predict(self, query_data):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      nb_queries = 100

      matches = 0
      for i in range(nb_queries):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'], 
                                                  self.Y: self.data['y'], 
                                                 self.Q: query_data['x'][i]})

        if query_data['y'] is not None:
          actual = query_data['y'][i]
          match = (hyp_val == actual)
          if match:
            matches += 1
    
      accuracy = matches / nb_queries

      return accuracy

  def hypvalues(self, query_data):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      pred_value = []
      for i in range(len(query_data)):
        hyp_val = sess.run(self.hyp, feed_dict = {self.X: self.data['x'], 
                                                  self.Y: self.data['y'], 
                                                 self.Q: query_data[i]})
        pred_value.append(hyp_val)

      return pred_value

#ucitavanja dataset-a i uzimanje samo poslednja dva feature-a
dataset = pd.read_csv('/data/social_network_ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#Podela na trening i test deo
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
nb_features = 2
nb_classes = 2
k=3
train_data = {'x': X_train, 'y': y_train}
knn = KNN(nb_features, nb_classes, train_data, k)
accuracy = knn.predict({'x': X_test, 'y': y_test})
print('Test set accuracy: ', accuracy)

#crtanje trazenog grafika
x1, x2 = np.meshgrid(np.arange(min(train_data['x'][:, 0]), max(train_data['x'][:, 0]), 
                                 5),
                       np.arange(min(train_data['x'][:, 1]), max(train_data['x'][:, 1]), 
                                 5000))

x_feed = np.vstack((x1.flatten(), x2.flatten())).T
pred_values = np.array(knn.hypvalues(x_feed))
pred_plot = pred_values.reshape([x1.shape[0], x1.shape[1]])



from matplotlib.colors import LinearSegmentedColormap
classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', 
                                                   ['lightblue', 
                                                    'lightyellow'])

plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

idxs_0 = train_data['y'] == 0
idxs_1 = train_data['y'] == 1
plt.scatter(train_data['x'][idxs_0, 0], train_data['x'][idxs_0, 1], c='b', 
            edgecolors='k', label='Klasa 0')
plt.scatter(train_data['x'][idxs_1, 0], train_data['x'][idxs_1, 1], c='y', 
            edgecolors='k', label='Klasa 1')


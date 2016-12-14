import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# We need to normalize the data because we're using Gradient Descent
# in order to bring all of the values to the same range
def normalize(data):
    return (data - data.mean()) / data.std()


# Parameters
learning_rate = 0.1  # Used for steps in gradient descent
training_epochs = 100  # How many times to iterate over training data
display_step = 20  # Show the calculated weights

# load the data
df = pd.read_csv('data.csv', header=None)
train_X = normalize(df[0])
train_Y = normalize(df[1])
n_samples = train_X.shape[0]

# plot the data to see what it looks like
plt.scatter(train_X, train_Y)
plt.show()

# Create TF graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Create the model
Yhat = tf.add( tf.mul(X, W), b )

# Cost function using mean squared error
cost_fn = tf.reduce_sum( tf.pow(Yhat - Y, 2) ) / (2*n_samples)

# Use gradient descent for optimisation
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_fn)

# Initialize all variables
init = tf.global_variables_initializer()

# Start the session
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Show weights calculated so far
        if (epoch+1) % display_step == 0:
            c = sess.run(cost_fn, feed_dict={X: train_X, Y:train_Y})
            print("Epoch: {}, cost = {}, W = {}, b = {}" \
                    .format(epoch+1, c, sess.run(W), sess.run(b)))

    print("Optimization Finished!")
    training_cost = sess.run(cost_fn, feed_dict={X: train_X, Y: train_Y})
    print("Training cost = {}, W = {}, b = {}" \
            .format(training_cost, sess.run(W), sess.run(b)))

    # Graphic display
    plt.plot(train_X, train_Y, "ro", label="Original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()

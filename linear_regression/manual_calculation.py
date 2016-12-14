import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('data.csv', header=None)
X = df[0]
Y = df[1]

# plot the data to see what it looks like
plt.scatter(X, Y)
plt.show()

# calculate values which are used more than once
Xsum = X.sum()
Xmean = X.mean()
Ymean = Y.mean()

# calculate the common denominator
denominator = X.dot(X) - Xmean * Xsum

# finally calculate the weights
W = ( X.dot(Y) - Ymean*Xsum ) / denominator
b = ( Ymean * X.dot(X) - Xmean * X.dot(Y) ) / denominator

print('Weights are: W = {}, b = {}'.format(W, b))

# calculate the predicted Y
Yhat = W*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print('R2 is: {}'.format(r2))

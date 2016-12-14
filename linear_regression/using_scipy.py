import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# load the data
df = pd.read_csv('data.csv', header=None)
X = df[0]
Y = df[1]

# plot the data to see what it looks like
plt.scatter(X, Y)
plt.show()

W, b, r_value, p_value, stderr = linregress(X, Y)

print('Weights are: W = {}, b = {}'.format(W, b))

# calculate the predicted Y
Yhat = W*X + b

# let's plot everything together to make sure it worked
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# determine how good the model is by computing the r-squared
r2 = r_value * r_value

print('R2 is: {}'.format(r2))

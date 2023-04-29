import numpy as np
import matplotlib.pyplot as plt

xs = np.array([0, -1, -2.5, -3, -4])
ys = np.array([4, 2, 0, -2, -4])
n = xs.shape[0]

xy_sum = (xs * ys).sum()
x_sum = xs.sum()
y_sum = ys.sum()
xsq_sum = (xs * xs).sum()
slope = (n * xy_sum - x_sum * y_sum) / (n * xsq_sum - x_sum * x_sum)
b = (y_sum - slope * x_sum) / n

fig, ax = plt.subplots()
ax.scatter(xs, ys)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axline((0, b), slope=slope, c='#ff0000')
plt.show()

import numpy as np

res = np.array([[1.964, 0.508, 0.688, 0.907, 0.625, 0.693, 0.682, 0.792],
                [5.126, 0.679, 0.995, 1.342, 0.956, 0.827, 0.949, 1.067],
                [1.295, 2.981, 2.538, 3.911, 1.500, 2.717, 1.847, 1.027],
                [0.929, 4.544, 2.209, 3.175, 2.188, 3.322, 1.322, 0.481],
                [1.004, 1.557, 2.868, 4.126, 1.091, 1.611, 1.890, 0.878],
                [1.001, 1.613, 4.318, 4.860, 1.377, 1.384, 1.668, 0.682],
                [0.840, 1.423, 2.042, 3.295, 0.799, 1.265, 1.562, 0.541],
                [1.125, 2.851, 5.331, 9.352, 1.605, 2.245, 1.869, 0.717],
                [0.603, 0.714, 0.736, 1.043, 0.629, 0.769, 0.715, 0.369],
                [1.213, 2.366, 1.365, 2.144, 3.265, 1.995, 1.267, 0.562],
                [0.610, 1.033, 1.113, 1.664, 0.675, 1.021, 0.711, 0.340],
                [0.909, 2.984, 1.497, 1.955, 1.926, 5.233, 1.132, 0.491],
                [0.791, 0.683, 0.616, 0.874, 0.693, 0.753, 3.409, 0.610],
                [0.602, 0.674, 0.780, 1.068, 0.676, 0.792, 5.997, 0.415],
                [2.099, 0.717, 1.380, 1.943, 0.904, 0.818, 1.121, 2.633],
                [2.650, 0.798, 1.308, 1.785, 0.779, 1.023, 1.778, 5.770]])

res_interfacegan = res[0::2]
res_ours = res[1::2]

res_interfacegan_ = res_interfacegan / np.diag(res_interfacegan)
res_ours_ = res_ours / np.diag(res_ours)

print(res_interfacegan_)
print()
print(res_ours_)
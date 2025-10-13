import numpy as np
import warnings

#14
# rng = np.random.default_rng()
# rand_vector = rng.integers(10, size=(30))
# print(rand_vector, rand_vector.mean())

#15
# arr = np.ones((5, 5))
# arr[1: -1, 1: -1] = 0
# print(arr)

#16
# arr = np.ones((5, 5))
# rows, cols = arr.shape
# border_arr = np.zeros((rows + 2, cols + 2))
# border_arr[1: -1, 1: -1] = arr
# print(arr, "\n\n", border_arr)

#17
# print(0 * np.nan)
# print(np.nan == np.nan)
# print(np.inf > np.nan)
# print(np.nan - np.nan)
# print(np.nan in set([np.nan]))
# print(0.3 == 3 * 0.1)

#18
# matrix = np.zeros((5, 5))
# for i in range(1, 5):
#     matrix[i, i - 1] = i
# print(matrix)

#19
# matrix = np.zeros((8, 8))
# for i in range(8):
#     for j in range(8):
#         if ((i + j) % 2 == 0):
#             matrix[i, j] = 1
# print(matrix)

#20
# arr = np.zeros((6, 7, 8))
# shape = arr.shape
# x = 99 // (shape[1] * shape[2])
# y = (99 % (shape[1] * shape[2])) // shape[2]
# z = (99 % (shape[1]* shape[2])) % shape[2]
# print(f'x: {x}, y: {y}, z: {z}')

#21
# pattern = np.array([[1, 0], [0, 1]])
# matrix = np.tile(pattern, (4, 4))
# print(matrix)

#22
# np.random.seed(42)
# random_matrix = np.random.randn(5, 5) * 10
# normalized = random_matrix / np.sum(np.abs(random_matrix), axis=1, keepdims=True)
# print(normalized)

#23
# rgba_dtype = np.dtype([
#     ('R', 'u1'),
#     ('G', 'u1'),
#     ('B', 'u1'),
#     ('A', 'u1')
# ])
# colors = np.array([
#     (255, 0, 0, 255),
#     (0, 255, 0, 255)
# ], dtype=rgba_dtype)

#24
# matrix_5x3 = np.random.randint(1, 10, size=(5, 3))
# matrix_3x2 = np.random.randint(1, 10, size=(3, 2))
# print(np.dot(matrix_5x3, matrix_3x2))

#25
# arr = np.array([1, 5, 8, 3, 10, 2, 7, 4, 9, 6])
# print(arr)
# arr[(arr >= 3) & (arr <= 8)] *= -1
# print(arr)

#26
# print(sum(range(5),-1))
# print(np.sum(range(5),-1))

#27
# Z = np.array([1, 2, 3, 4])

# print("Z =", Z)
# print("\nZ**Z:", Z**Z)
# print("Z << Z >> 2:", Z << Z >> 2)
# print("Z <- Z:", Z < -Z)
# print("1j*Z:", 1j*Z)                
# print("Z/1/1:", Z/1/1)                
# print("Z<Z>Z:", Z<Z>Z)

#28
# print("np.array(0) / np.array(0):", np.array(0) / np.array(0))
# print("np.array(0) // np.array(0):", np.array(0) // np.array(0))
# print("np.array([np.nan]).astype(int).astype(float):", np.array([np.nan]).astype(int).astype(float))

#29
# arr = np.array([0, 0.1, 0.5, 0.9, 1, -0.1, -0.5, -0.9, -1, 2.3, -2.3])
# print(np.sign(arr) * np.ceil(np.abs(arr)))

#30
# arr1 = np.array([1, 2, 3, 4, 5, 2])
# arr2 = np.array([3, 4, 5, 6, 7, 4])
# print(np.array(list(set(arr1) & set(arr2))))

#31
# warnings.filterwarnings('ignore')
# result = np.array([1, 2, 3]) / 0

#32
# print(np.sqrt(-1) == np.emath.sqrt(-1))

#33
# print('Вчера: ', np.datetime64('today') - np.timedelta64(1, 'D'))
# print('Сегодня: ', np.datetime64('today'))
# print('Завтра: ', np.datetime64('today') + np.timedelta64(1, 'D'))

#34
# print(np.arange(np.datetime64('2016-07-01'), np.datetime64('2016-08-01'), dtype='datetime64'))

#35
# A = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
# B = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
# A += B
# B = -(A - B) / 2
# A *= B
# print(A)

#36
# np.random.seed(42)
# random_array = np.random.uniform(0, 10, size=8)
# print(random_array)
# print(np.trunc(random_array))
# print(random_array.astype(int))
# print(random_array - (random_array % 1))

#37
# matrix = np.tile(np.arange(5), (5, 1))
# print(matrix)

#38
# rng = np.random.default_rng()
# arr = np.array(rng.random(10))
# print(arr)

#39
# vector = np.linspace(0, 1, endpoint=False)[1:]
# print(vector)

#40
# vector = np.random.random(10)
# vector.sort()
# print(vector)

#41
# arr = np.array([1, 8, 9, 4, 5])
# arr = np.add.reduce(arr)
# print(arr)

#42
# arr1 = np.random.random(5)
# arr2 = np.random.random(5)
# print(arr1, arr2)
# print(np.array_equal(arr1, arr2))

#43
# arr = np.array([2, 4, 1])
# arr.flags.writeable = False

#44
# z = np.random.random((10,2))
# x, y = z[:,0], z[:,1]
# r = np.sqrt( x**2 + y**2)
# t = np.arctan2(y, x)
# print(r, t)

#45
# z = np.random.random(10)
# print(z)
# z[z.argmax()] = 0
# print(z)

#46
# z = np.zeros((5, 5), [('x', float), ('y', float)])
# z['x'], z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
# print(z)

#47
# X = np.arange(8)
# Y = X + 0.5
# C = 1.0 / np.subtract.outer(X, Y)
# print(C)

#48
# for dtype in [np.int8, np.int32, np.float32, np.float64]:
#     try:
#         print(np.iinfo(dtype).min, np.iinfo(dtype).max)
#     except:
#         print(np.finfo(dtype).min, np.finfo(dtype).max)

#49
# Z = np.random.random(10)
# np.set_printoptions(threshold=np.inf)
# print(Z)

#50
# Z = np.random.random(10)
# print(Z)
# value = 0.5
# print(Z[(np.abs(Z - value)).argmin()])

#51
# Z = np.zeros(10, [('position', [('x', float), ('y', float)]), ('color', [('r', float), ('g', float), ('b', float)])])
# print(Z)

#52
# Z = np.random.random((5,2))
# D = np.sqrt(((Z[:,np.newaxis,:] - Z[np.newaxis,:,:]) ** 2).sum(axis=2))
# print(Z, '\n\n', D)

#53
# arr = np.array([1.5, 2.7, 3.1, 4.9], dtype=np.float32)
# print(arr)
# arr = arr.astype(np.int32, copy=False)
# print(arr)

#54
# arr = np.genfromtxt('text.txt', delimiter=',', filling_values=0)
# print(arr)

#55
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# for i, val in np.ndenumerate(arr):
#     print(i, val)

#56
# x = np.linspace(-3, 3, 4)
# y = np.linspace(-3, 3, 4)
# X, Y = np.meshgrid(x, y)
# Z = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
# print(Z)

#57
# arr = np.zeros((5, 5), dtype=int)
# p = 4
# value = 1
# arr.flat[np.random.choice(arr.shape[0] * arr.shape[1], p, replace=False)] = value
# print(arr)

#58
# matrix = np.array([[1, 9, 3],[4, 5, 7],[4, 8, 3]], dtype=float)
# row_means = matrix.mean(axis=1, keepdims=True)
# print(matrix - row_means)

#59
# arr = np.array([[3, 2, 1],[2, 5, 4],[9, 8, 7]])
# n = 0
# sorted_arr = arr[arr[:, n].argsort()]
# print(sorted_arr)

#60
# arr = np.array([[1, 0, 3, 0],[0, 0, 4, 0],[2, 0, 5, 0]])
# print(np.any(np.all(arr == 0, axis=0)))

#61
# arr_2d = np.array([[1, 5, 9],[2, 8, 3],[7, 4, 6]])
# target = 10
# print(arr_2d.flat[np.abs(arr_2d - target).argmin()])

#62
# a = np.array([[1, 2, 3]])
# b = np.array([[4], [5], [6]])
# result = np.empty((3, 3))
# it = np.nditer([a, b, result], flags=['multi_index'], op_flags=[['readonly'], ['readonly'], ['writeonly']])
# for x, y, z in it:
#     i, j = it.multi_index
#     z[...] = a[0, j] + b[i, 0]
# print(result)

#63
# class NamedArray:
#     def __init__(self, name, array):
#         self.name = name
#         self.array = np.array(array)
# test = NamedArray('arr', [1, 2, 3])
# print(f'{test.name}: {test.array}')

#64
# Z = np.array([0, 0, 0, 0, 0])
# I = np.array([0, 1, 1, 3, 3, 4])
# np.add.at(Z, I, 1)
# print(Z)

#65
# Z = np.array([1.5, 2.0, 0.8, 1.2, 3.1])
# I = np.array([0, 1, 0, 2, 1])
# sum_Z_I = np.zeros(np.max(I) + 1)
# np.add.at(sum_Z_I, I, Z)
# print("Суммарное время по серверам:", sum_Z_I)

#66
# w, h = 100, 100
# image = np.random.randint(0, 256, (w, h, 3), dtype=np.uint8)
# colors = image.reshape(-1, 3)
# unique_colors = np.unique(colors, axis=0)
# print("Размер изображения:", image.shape)
# print("Количество уникальных цветов:", len(unique_colors))
# import time
# start = time.time()
# unique_count = len(np.unique(image.reshape(-1, 3), axis=0))
# end = time.time()
# print(f"Время выполнения: {end-start:.6f} секунд")

#67
# arr_4d = np.random.rand(5, 3, 10, 8)
# sum_last_two = np.sum(arr_4d, axis=(-2, -1))
# print("Результат:")
# print(sum_last_two)

#68
# Z = np.array([1.5, 2.0, 0.8, 1.2, 3.1, 2.5])
# I = np.array([0, 1, 0, 2, 1, 0])
# unique_I = np.unique(I)
# mean_Z_I = np.full(np.max(I) + 1, np.nan)
# for server in unique_I:
#     mask = I == server
#     mean_Z_I[server] = np.mean(Z[mask])
# print(mean_Z_I)

#69
# A = np.array([[1, 2, 3],[4, 5, 6]])
# B = np.array([[7, 8],[9, 10],[11, 12]])
# diag_dot = np.einsum('ij,ji->i', A, B)
# print("Диагональ через einsum:", diag_dot)
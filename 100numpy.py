# 1
# import numpy as np

#2
# print(f'Версия: {np.version}')
# print('Конфигурация:\n')
#
# np.show_config()
#
# #3
# zero_vector = np.zeros(10)
# print(zero_vector)
#
# #5
# help(np.add)
#
# #6
# null_vector = np.zeros(10)
# null_vector[4] = 1
# print(null_vector)
#
# #7
# range_vector = np.arange(10, 50)
# print(range_vector)
#
# #8
# reversed_vector = np.flip(range_vector)
# print(reversed_vector)
#
# #9
# matrix3x3 = np.arange(9).reshape(3, 3)
# print(matrix3x3)
#
# #10
# vector = np.array([1, 2, 0, 0, 4, 0])
# indeces = np.nonzero(vector)
# print(vector, indeces)
#
# #11
# identity_matrix = np.ones((3, 3))
# print(identity_matrix)
#
# #12
# rng = np.random.default_rng()
# array_3x3x3 = rng.integers(10, size=(3, 3, 3))
# print(array_3x3x3)

#13
# rng = np.random.default_rng()
# array_10x10 = rng.integers(20, size=(10, 10))
# print(array_10x10)
# print('Min: ', array_10x10.min())
# print('Min: ', array_10x10.max())

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
# print(diag_dot)+

#70
# Z = np.array([1, 2, 3, 4, 5])
# result = np.zeros(len(Z) + (len(Z)-1)*3, dtype=Z.dtype)
# result[::4] = Z
# print(result)

#72
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(a)
# a[[0, 2]] = a[[2, 0]]
# print(a)

#73
# triangles = np.random.randint(0, 100, (10, 3))
# segments = np.vstack([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]])
# segments = np.sort(segments, axis=1)
# unique_segments = np.unique(segments, axis=0)
# print(unique_segments)

#74
# C = np.array([0, 2, 1, 3, 0, 1])
# A = np.repeat(np.arange(len(C)), C)
# print(A)

#75
# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# window_size = 3
# moving_avg = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
# print(arr)
# print(moving_avg)

#76
# Z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# window_size = 3
# result = np.array([Z[i:i+window_size] for i in range(len(Z) - window_size + 1)])
# print(Z)
# print(result)

#77
#77.1
# bool_arr = np.array([True, False, True, False])
# np.logical_not(bool_arr, out=bool_arr)
# print(bool_arr)

#77.2
# float_arr = np.array([1.5, -2.5, 3.0, -4.0])
# np.negative(float_arr, out=float_arr)
# print(float_arr)

#78
# P0 = np.array([[0,0], [1,1], [2,0]])
# P1 = np.array([[1,0], [2,2], [3,1]])
# p = np.array([1, 1])
# distances = np.abs(np.cross(P1-P0, p-P0) / np.linalg.norm(P1-P0))
# print(distances)

#79
# P0 = np.array([[0, 0], [1, 0], [0, 1]]) 
# P1 = np.array([[1, 1], [2, 1], [1, 2]])
# P = np.array([[0.5, 0.5], [1, 1], [0, 0]])
# distances = np.abs(np.cross(P1 - P0, P[:, np.newaxis] - P0)) / np.linalg.norm(P1 - P0, axis=1)
# print(distances)

#80
# def extract_subpart(arr, center, shape, fill_value=0):
#     result = np.full(shape, fill_value, dtype=arr.dtype)
    
#     start_i = max(0, center[0] - shape[0]//2)
#     end_i = min(arr.shape[0], center[0] + shape[0]//2 + 1)
#     start_j = max(0, center[1] - shape[1]//2)
#     end_j = min(arr.shape[1], center[1] + shape[1]//2 + 1)
    
#     res_start_i = max(0, shape[0]//2 - center[0])
#     res_end_i = res_start_i + (end_i - start_i)
#     res_start_j = max(0, shape[1]//2 - center[1])
#     res_end_j = res_start_j + (end_j - start_j)
    
#     result[res_start_i:res_end_i, res_start_j:res_end_j] = \
#         arr[start_i:end_i, start_j:end_j]
    
#     return result

# arr = np.arange(25).reshape(5, 5)
# subpart = extract_subpart(arr, (2, 2), (3, 3))
# print(arr)
# print(subpart)

#81
# Z = np.arange(1, 15)
# window_size = 4
# R = Z[np.arange(window_size) + np.arange(len(Z) - window_size + 1).reshape(-1, 1)]
# print(R)

#82
# matrix = np.random.random((4, 4))
# rank = np.linalg.matrix_rank(matrix)
# print(rank)

#83
# Z = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
# values, counts = np.unique(Z, return_counts=True)
# most_common = values[np.argmax(counts)]
# print(most_common)

#84
# def extract_3x3_blocks(matrix):
#     blocks = []
#     for i in range(matrix.shape[0] - 2):
#         for j in range(matrix.shape[1] - 2):
#             blocks.append(matrix[i:i+3, j:j+3])
#     return np.array(blocks)

# matrix_10x10 = np.random.random((10, 10))
# blocks_3x3 = extract_3x3_blocks(matrix_10x10)
# print(blocks_3x3)

#85
# class SymmetricArray(np.ndarray):
#     def __setitem__(self, index, value):
#         i, j = index
#         super().__setitem__((i, j), value)
#         super().__setitem__((j, i), value)

# base = np.zeros((3, 3))
# symmetric = base.view(SymmetricArray)
# symmetric[0, 1] = 5
# print(symmetric)

#86
# p, n = 3, 4
# matrices = np.random.random((p, n, n))
# vectors = np.random.random((p, n, 1))
# result = np.sum(matrices @ vectors, axis=0)
# print(result.shape)

#87
# matrix_16x16 = np.arange(256).reshape(16, 16)
# h, w = matrix_16x16.shape
# bh, bw = 4, 4
# block_sums = matrix_16x16.reshape(h//bh, bh, w//bw, bw).sum(axis=(1, 3))
# print(f"\n4x4 block sums:")
# print(block_sums)

#88
# def game_of_life(grid, steps=1):
#     for _ in range(steps):
#         neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
#                        for i in (-1, 0, 1) for j in (-1, 0, 1)
#                        if (i != 0 or j != 0))
#         birth = (neighbors == 3) & (grid == 0)
#         survive = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
#         grid = np.where(birth | survive, 1, 0)

#     return grid
# initial = np.zeros((10, 10))
# initial[1, 2] = 1
# initial[2, 3] = 1
# initial[3, 1:4] = 1

# next_gen = game_of_life(initial)
# print(next_gen)

#89
# def n_largest(arr, n):
#     if arr.ndim == 1:
#         return arr[np.argpartition(arr, -n)[-n:]]
#     else:
#         flat_indices = np.argpartition(arr.ravel(), -n)[-n:]
#         return np.unravel_index(flat_indices, arr.shape), arr.flat[flat_indices]

# large_array = np.random.random(1000)
# n_largest_vals = n_largest(large_array, 5)
# print(n_largest_vals)

#90
# def cartesian_product(arrays):
#     arrays = [np.asarray(arr) for arr in arrays]
#     shape = [len(arr) for arr in arrays]
#     ix = np.indices(shape)
#     ix = ix.reshape(len(arrays), -1).T
#     result = np.empty(ix.shape)
#     for n, arr in enumerate(arrays):
#         result[:, n] = arrays[n][ix[:, n]]
#     return result

# A = [1, 2, 3]
# B = [4, 5]
# cartesian = cartesian_product([A, B])
# print(cartesian)

#91
# regular_array = np.array([(1, 2.0, 'Hello'), (2, 3.0, 'World')], dtype=[('x', 'i4'), ('y', 'f4'), ('z', 'U10')])
# record_array = np.rec.array(regular_array)
# print(record_array)
# print(f"Field access: x={record_array.x}, y={record_array.y}")

#92
# Z = np.array([1, 2, 3, 4, 5])
# print(f"\nZ = {Z}")
# print("**: ", Z ** 3)
# print("np.power: ", np.power(Z, 3))
# print("multiplication:", Z * Z * Z)

#93
# def rows_containing_all_B(A, B):
#     result = []
#     for i, row_a in enumerate(A):
#         contains_all = True
#         for row_b in B:
#             if not any(elem in row_a for elem in row_b):
#                 contains_all = False
#                 break
#         if contains_all:
#             result.append(i)
#     return np.array(result)

# A = np.random.randint(0, 10, (8, 3))
# B = np.array([[4, 3], [1, 2]])
# matching_rows = rows_containing_all_B(A, B)
# print(matching_rows)

#94
# matrix_10x3 = np.random.randint(0, 3, (10, 3))
# unequal = matrix_10x3[np.any(matrix_10x3 != matrix_10x3[:, [0]], axis=1)]
# print(unequal)

#95
# def int_to_binary_matrix(arr):
#     max_val = np.max(arr)
#     num_bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
#     binary = (((arr[:, None] & (1 << np.arange(num_bits-1, -1, -1))) > 0).astype(int))
#     return binary

# arr = np.array([0, 1, 8])
# binary_matrix = int_to_binary_matrix(arr)
# print(binary_matrix)

#97
# A = np.array([1, 2, 3])
# B = np.array([4, 5, 6])

# print("np.inner:", np.inner(A, B))
# print("einsum:", np.einsum('i,i', A, B))

# print("np.outer:", np.outer(A, B))
# print("einsum:", np.einsum('i,j->ij', A, B))

# print("np.sum:", np.sum(A))
# print("einsum:", np.einsum('i->', A))

# print("A * B:", A * B)
# print("einsum:", np.einsum('i,i->i', A, B))

#98
# def equidistant_samples(X, Y, n_samples):
#     distances = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
#     cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
#     total_dist = cumulative_dist[-1]
#     sample_dists = np.linspace(0, total_dist, n_samples)
#     sample_X = np.interp(sample_dists, cumulative_dist, X)
#     sample_Y = np.interp(sample_dists, cumulative_dist, Y)
#     return sample_X, sample_Y

# X = np.array([0, 1, 2, 3, 4])
# Y = np.array([0, 1, 0, 1, 0])
# samples_X, samples_Y = equidistant_samples(X, Y, 10)
# print(f"X: {samples_X}")
# print(f"Y: {samples_Y}")

#99
# def multinomial_rows(X, n):
#     is_integer = np.all(X == np.floor(X), axis=1)
#     sums_to_n = np.sum(X, axis=1) == n
#     return X[is_integer & sums_to_n]

# X = np.array([[1, 2, 1], 
#               [2.5, 1, 1.5], 
#               [3, 0, 1], 
#               [1, 1, 2]])
# n = 4
# is_integer = np.all(X == np.floor(X), axis=1)
# sums_to_n = np.sum(X, axis=1) == n
# X[is_integer & sums_to_n]
# multinomial = X[is_integer & sums_to_n]
# print(multinomial)
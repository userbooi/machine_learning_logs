import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([
    [3, 4, 5],
    [7, 2, 1],
    [9, 3, 2]
])

arr3 = np.array([
    [2],
    [4],
    [5],
    [3]
])
arr4 = np.array([
    [1, 3, 4, 6],
    [4, 2, 1, 2],
    [10, 5, 6, 9]
])

print(np.dot(arr2, arr1))
print(np.dot(arr4, arr3))

arr5 = np.array([
    [3, 2, 1],
    [6, 5, 7],
    [9, 10, 11],
])
arr6 = np.array([100, 200, 300])
arr7 = np.array([
    [100],
    [200],
    [300],
])
print(np.add(arr5, arr6))
print(np.add(arr5, arr7))

print()

print(arr4.reshape((3, 2, 2)))
print(arr4.shape[0], arr4.shape[1])

print()

row_norms_arr5 = np.linalg.norm(arr5, ord=2, axis=1, keepdims=True)
print(row_norms_arr5)

norm_arr5 = arr5/row_norms_arr5
print(norm_arr5)

sum_no_keepdims = (norm_arr5**2).sum(axis=1)
sum_keepdims = (norm_arr5**2).sum(axis=1, keepdims=True)
print(sum_no_keepdims)
print(sum_keepdims)

# check the dimensions of rank 0 arrays vs 1 row rank 1 arrays
arr8 = np.array([1, 2, 3, 4, 5, 6])
arr9 = np.array([[1, 2, 3,4 ,5 ,6]])
print(arr8.shape)
print(arr9.shape)


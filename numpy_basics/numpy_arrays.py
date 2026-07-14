import numpy as np

'''
NumPy arrays only contain same elements (it is not a list)

number of dimensions is called an array's RANK
size of the array along each dimension is called its SHAPE
'''

# ================numpy arrays are created from python lists================
# arr1 = np.array([1, 2, 3, 4]) # 1D
# arr2 = np.array([[1, 2, 3], [4, 5, 6]]) # 2D
# arr3 = np.array((1, 2, 3)) # from tuples
#
# when the types are mixed, the array will convert everything to the type that fits all
# int + float = float
# float + string = string
# arr4 = np.array([1.0, 2.3, 3, "ah hell naw"])
#
# cannot create an array with different dimentions like this:
# arr5 = np.array([[1, 2, 3, 2], [2, 3, 2], [3]])
# print(arr5)
#
# print(f"{arr1}\n{arr2}\n{arr3}")

#========================create arrays using builtin methods========================
# zeros_array = np.zeros((2, 3))
# print(zeros_array)
#
# #                        face row col
# zeros_array_3D = np.zeros((3, 4, 2))
# print(zeros_array_3D)
#
# ones_array = np.ones((3, 2))
# print(ones_array)
#
# # Generates an array with values in a specified range.
# #                     start end skip
# range_array1 = np.arange(0, 10, 2)
# print(range_array1)
#
# range_array2 = np.arange(0, 10, 3)
# print(range_array2)
#
# linear_array = np.linspace(0, 1, 5)
# print(linear_array)
#
# # generate an array starting at 0 to 10, excluding 10
# consecutive = np.arange(10)
# print(consecutive)
#
# # reshape an array to the specified shape
# # the new dimensions must match the total number of elements
# reshaped1 = np.reshape(np.arange(6), (2, 3))
# print(reshaped1)
#
# reshaped2 = np.reshape(np.array([[1, 3, 2], [3, 2, 3], [4, 3, 2], [10, 12, 11]], dtype=np.float64), (2, 6))
# print(reshaped2)
#
# # turns a multi-deminsional array into 1D
# flattened = reshaped1.flatten()
# print(flattened)
#
# # concatenate two arrays (rank must be the same and the shape along a dimension)
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])
# c = np.array([[5, 6]])
# concatenated1 = np.concatenate((a, b))
# print(concatenated1)
#
# concatenated2 = np.concatenate((a, c))
# print(concatenated2)

# ============================accessing arrays==============================
# arr = np.array([
#     [-1, 2, 0, 4],
#     [4, -0.5, 6, 0],
#     [2.6, 0, 7, 8],
#     [3, -7, 4, 2.0]
# ])
#
# # :: means skip
# #          row  col
# arr2 = arr[:2, ::2]
# print("First 2 rows and alternate columns:\n", arr2)
#
# # slicing rows, then slice columns
# arr3 = arr[1:, 1:3]
# print("rows 1 to the end and cols 1 to 2:\n", arr3)
#
# #            row number     col number
# arr4 = arr[[1, 1, 0, 3], [3, 2, 1, 0]]
# print("Selected elements:", arr4)
#
# arr5 = arr[3]
# print("row 3:\n", arr5)
#
# ele = arr[2][1]
# print("element at (2, 1):\n", ele)

# ============================basic operations==========================
# a = np.array([[1, 2],
#               [3, 4]])
#
# b = np.array([[4, 3],
#               [2, 1]])
#
# print("Adding 1 to every element:\n", a + 1)
# print("Subtracting 2 from each element:\n", b - 2)
# print("Sum of all array elements:", a.sum())
# print("Array sum:\n", a + b)

#=============================data types===============================
# x = np.array([1, 2])
# print(x.dtype)
#
# x = np.array([1.0, 2.0])
# print(x.dtype)
#
# x = np.array([1.5, 2.2], dtype=np.int64) # force type
# print(x)
# print(x.dtype)

#==========================more operations==============================
arr1 = np.array([[4, 7], [2, 6]], dtype=np.float64)
arr2 = np.array([[3, 6], [2, 8]], dtype=np.float64)
arr3 = np.array([[4, 2, 3], [3, 2, 10]], dtype=np.float64)

# print(np.add(arr1, arr2)) # cannot add with different shapes (size)
# print(np.sum(arr1))
# print(np.sqrt(arr1))
# print(arr1.mean())
# print(arr1 * arr2)
# # transpose (swap the columns and rows)
# print(arr1.T)
# print(arr3.T)

# dot products
print(np.dot(arr1, arr2))
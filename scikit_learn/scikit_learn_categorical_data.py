from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

'''
ML algorithms work with numerical inputs. Categorical (text) data must be converted into numbers
(enumeration) in order for the algorithms to work.

the Iris dataset is already encoded. This is why the target array is made up of 0, 1, and 2

Another encryption is OneHot Encoding, which turns the categories into binary instead of real numbers
OneHot encoder must be reshaped into a 2D array
'''

#==========================handling categorical data==============================
categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird'] # more like targets

encoder = LabelEncoder()

encoded_feature = encoder.fit_transform(categorical_feature)

print("Encoded feature:", encoded_feature)

#===============================one hot encoding===================================
categorical_feature2 = ['cat', 'dog', 'dog', 'cat', 'bird']
# the -1 means automatic. Numpy will calculate the correct dimension (ONLY USABLE FOR ONE DIMENSION)
# reshape the feature into a 2D array with 5 rows and each row containing a category
categorical_feature2 = np.array(categorical_feature2).reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)

encoded_feature = encoder.fit_transform(categorical_feature2)

print("\nEncoded feature with OneHot Encoding:\n", encoded_feature)

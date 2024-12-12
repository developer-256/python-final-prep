import numpy as np

# 1.	Create a 3D array of shape (2, 3, 4) filled with random integers between 10 and 50.
# o	Print the shape of the array.
# o	Access and display all elements of the first 2D slice along the first axis.
array_3d = np.random.randint(10, 50, (2, 3, 4))
print("Shape of the array:", array_3d.shape)
print("First 2D slice along the first axis:")
print(array_3d[0])


# 2.	Create a 2D array of shape (4, 4) filled with ones. Replace the diagonal elements with values [5, 10, 15, 20].
array_2d = np.ones((4, 4))
np.fill_diagonal(array_2d, [5, 10, 15, 20])
print("Modified 2D array with diagonal values:")
print(array_2d)


# Question 1: NumPy (10 Marks)
# 1.	Generate an array of 30 random numbers between 0 and 1. Reshape it into a matrix of shape (5, 6).
# o	Calculate the sum of all rows and columns.
random_array = np.random.rand(30).reshape(5, 6)
print("5x6 Matrix:")
print(random_array)
print("Sum of all rows:", random_array.sum(axis=1))
print("Sum of all columns:", random_array.sum(axis=0))


# 2.	Create two arrays:
# o	arr1 = [1, 2, 3, 4] and arr2 = [5, 6, 7, 8] Perform the following operations:
# o	Element-wise addition
# o	Element-wise multiplication
# o	Dot product
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
print("Element-wise addition:", arr1 + arr2)
print("Element-wise multiplication:", arr1 * arr2)
print("Dot product:", np.dot(arr1, arr2))


# 4.	Create a 1D array of numbers from 1 to 12, then reshape it into a 3x4 matrix. 
reshaped_matrix = np.arange(1, 13).reshape(3, 4)
print("3x4 Matrix:")
print(reshaped_matrix)



# 5.	Create a 2D array of shape (3, 5) filled with zeros, then replace the second row with ones. 
array_zeros = np.zeros((3, 5))
array_zeros[1] = 1
print("Matrix with modified second row:")
print(array_zeros)



# 6.	Perform element-wise multiplication of two arrays: [1, 2, 3] and [4, 5, 6]. 
result_multiplication = np.array([1, 2, 3]) * np.array([4, 5, 6])
print("Element-wise multiplication:", result_multiplication)




# Part 1: Create a 1D array of 50 equally spaced numbers between 0 and 10
array_1d = np.linspace(0, 10, 50)

# Part 2: Reshape it into a 2D array of shape (10, 5)
array_2d = array_1d.reshape(10, 5)

# Part 3: Extract all elements that are greater than 5 and less than 8
extracted_elements = array_2d[(array_2d > 5) & (array_2d < 8)]

print("Extracted elements:", extracted_elements)


# 1. Create two 3x3 matrices filled with random integers between 10 and 50.
# 2. Perform the following operations:
# o Element-wise addition and subtraction.
# o Calculate the determinant of both matrices.
# o Find the inverse of the second matrix.

# Part 1: Create two 3x3 matrices with random integers between 10 and 50
matrix1 = np.random.randint(10, 50, (3, 3))
matrix2 = np.random.randint(10, 50, (3, 3))

# Element-wise addition and subtraction
addition = matrix1 + matrix2
subtraction = matrix1 - matrix2

# Determinant of both matrices
det_matrix1 = np.linalg.det(matrix1)
det_matrix2 = np.linalg.det(matrix2)

# Inverse of the second matrix
if det_matrix2 != 0:
    inverse_matrix2 = np.linalg.inv(matrix2)
else:
    inverse_matrix2 = "Matrix 2 is not invertible."

print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Element-wise addition:\n", addition)
print("Element-wise subtraction:\n", subtraction)
print("Determinant of Matrix 1:", det_matrix1)
print("Determinant of Matrix 2:", det_matrix2)
print("Inverse of Matrix 2:\n", inverse_matrix2)




# Question 3: Advanced Array Operations (4 Marks)
# 1. Create a 2D NumPy array of shape (6, 6) where each element is the product of its row
# and column indices (1-based indexing).
# 2. Split the array into three equal parts along the rows.
# 3. Concatenate the parts back into a single array along the columns.

# Part 1: Create a 2D array where each element is the product of its row and column indices (1-based indexing)
array_2d = np.fromfunction(lambda i, j: (i + 1) * (j + 1), (6, 6), dtype=int)

# Part 2: Split the array into three equal parts along the rows
split_arrays = np.array_split(array_2d, 3, axis=0)

# Part 3: Concatenate the parts back into a single array along the columns
concatenated_array = np.concatenate(split_arrays, axis=1)

print("Original Array:\n", array_2d)
print("Concatenated Array:\n", concatenated_array)


# Question 4: Broadcasting and Statistical Analysis (5 Marks)
# 1. Create a 3D NumPy array of shape (3, 4, 5) filled with random integers between 1 and
# 100.
# 2. Subtract the mean of each 2D matrix (slice along the first axis) from all elements in that
# slice.
# 3. Calculate the following:
# o The overall standard deviation of the array.
# o The variance of each 2D slice.

# Part 1: Create a 3D array of shape (3, 4, 5) with random integers between 1 and 100
array_3d = np.random.randint(1, 100, (3, 4, 5))

# Part 2: Subtract the mean of each 2D slice
mean_subtracted = array_3d - array_3d.mean(axis=(1, 2), keepdims=True)

# Part 3: Calculate overall standard deviation and variance of each 2D slice
std_dev = array_3d.std()
variance_slices = array_3d.var(axis=(1, 2))

print("Original 3D Array:\n", array_3d)
print("Mean Subtracted Array:\n", mean_subtracted)
print("Overall Standard Deviation:", std_dev)
print("Variance of each 2D slice:", variance_slices)



# Question 5: Optimization Using NumPy (4 Marks)
# 1. Write a function find_closest(array, value) that takes a 1D NumPy array and a target
# value, and returns the element in the array closest to the target value.
# 2. Test the function with an array of random integers between 1 and 100 and a target value
# of 50.
# Part 1: Define the find_closest function
def find_closest(array, value):
    return array[np.abs(array - value).argmin()]

# Part 2: Test the function
test_array = np.random.randint(1, 100, 20)
target_value = 50
closest_value = find_closest(test_array, target_value)

print("Test Array:", test_array)
print("Target Value:", target_value)
print("Closest Value:", closest_value)





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pandas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
# Implement the following question in python using Pandas library.
# 1.	Create a DataFrame from the following dictionary:
# data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'Salary': [50000, 60000, 70000]}

# Task 1: Create a DataFrame and display it
# Display the DataFrame.
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'Salary': [50000, 60000, 70000]}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)




# 2.	Add a new column Department to the DataFrame with values ['HR', 'IT', 'Finance'].

# Task 2: Add a new column 'Department'
df['Department'] = ['HR', 'IT', 'Finance']
print("\nDataFrame with Department column:")
print(df)





# 3.	Save the DataFrame to a CSV file named employees.csv.

# Task 3: Save the DataFrame to a CSV file
df.to_csv('employees.csv', index=False)








# 1.	Create a Series from the following list: [10, 20, 30, 40, 50]. Display the Series and its index.

# Task 1: Create a Series and display it
series = pd.Series([10, 20, 30, 40, 50])
print("Series:")
print(series)
print("Index:", series.index)






# 2.	Create a DataFrame with the following data:
# Columns: ['Product', 'Price', 'Quantity']
# Data: [['Laptop', 80000, 5], ['Phone', 50000, 10], ['Tablet', 30000, 7]]
# Calculate the total value (Price × Quantity) for each product and add it as a new column Total.

# Task 2: Create a DataFrame and calculate the total value
products_data = {'Product': ['Laptop', 'Phone', 'Tablet'], 'Price': [80000, 50000, 30000], 'Quantity': [5, 10, 7]}
products_df = pd.DataFrame(products_data)
products_df['Total'] = products_df['Price'] * products_df['Quantity']
print("\nDataFrame with Total column:")
print(products_df)






# 3.	Load the DataFrame from a CSV file named sales.csv and display the first 5 rows.
sales_df = pd.read_csv('sales.csv')
print("\nFirst 5 rows of the sales DataFrame:")
print(sales_df.head(5))





# Question 1: Pandas (10 Marks)
# 1.	Create a DataFrame from the following dictionary:
# data = {'Country': ['Pakistan', 'India', 'China'], 
#         'Population': [230, 1400, 1441], 
#         'GDP': [376, 3229, 15461]}
# Task 1: Create a DataFrame and add GDP per Capita
countries_data = {'Country': ['Pakistan', 'India', 'China'], 
                  'Population': [230, 1400, 1441], 
                  'GDP': [376, 3229, 15461]}
countries_df = pd.DataFrame(countries_data)
countries_df['GDP per Capita'] = countries_df['GDP'] / countries_df['Population']
print("Countries DataFrame:")
print(countries_df)
# o	Display the DataFrame.
# o	Add a new column GDP per Capita calculated as GDP / Population.
# o	Save the DataFrame to a CSV file named countries.csv.
countries_df.to_csv('countries.csv', index=False)





# 2.	Load a CSV file named employees.csv into a DataFrame and perform the following:
# o	Display the first 3 rows.
employees_df = pd.read_csv('employees.csv')
print("\nFirst 3 rows of employees.csv:")
print(employees_df.head(3))





# o	Sort the DataFrame by the "Salary" column in descending order.
sorted_employees = employees_df.sort_values(by='Salary', ascending=False)
print("\nEmployees sorted by Salary (descending):")
print(sorted_employees)



# Question 1: Pandas (5 Marks)
# 1.	Create a Series from the list [100, 200, 300, 400, 500] with indices as the first five letters of the alphabet (['A', 'B', 'C', 'D', 'E']).
# o	Display the Series and its indices.
series = pd.Series([100, 200, 300, 400, 500], index=['A', 'B', 'C', 'D', 'E'])
print("Series with custom indices:")
print(series)
print("Indices:", series.index)




# 2.	Create a DataFrame with the following data:
# Columns: ['Product', 'Price', 'Stock']
# Data: [['Laptop', 80000, 10], ['Phone', 50000, 20], ['Tablet', 30000, 15]]
# o	Add a new column Total Value calculated as Price × Stock.
inventory_data = {'Product': ['Laptop', 'Phone', 'Tablet'], 
                  'Price': [80000, 50000, 30000], 
                  'Stock': [10, 20, 15]}
inventory_df = pd.DataFrame(inventory_data)
inventory_df['Total Value'] = inventory_df['Price'] * inventory_df['Stock']
print("\nDataFrame with Total Value column:")
print(inventory_df)



# o	filter rows where the Stock is greater than 10 and display the filtered DataFrame.
filtered_df = inventory_df[inventory_df['Stock'] > 10]
print("\nFiltered DataFrame where Stock > 10:")
print(filtered_df)





# 
data = {
    'Student': ['Ali', 'Sara', 'Ahmed', 'Zoya'],
    'Subject': ['Math', 'English', 'Science', 'History'],
    'Marks': [88, 75, 93, 68],
    'Grade': ['A', 'B', 'A', 'C']
}
df = pd.DataFrame(data)
print(df)




# 
# Add Pass/Fail column
df['Pass/Fail'] = df['Marks'].apply(lambda x: 'Pass' if x >= 70 else 'Fail')

# Group by Grade and calculate average marks
avg_marks = df.groupby('Grade')['Marks'].mean()
print(avg_marks)




data = {
    "Product": ["Laptop", "Tablet", "Smartphone", "Monitor", "Keyboard"],
    "Category": ["Electronics", "Electronics", "Electronics", "Electronics", "Accessories"],
    "Price": [80000, 30000, 20000, 15000, 5000],
    "Units_Sold": [5, 8, 15, 10, 50]
}
df = pd.DataFrame(data)

# Add Revenue column
df['Revenue'] = df['Price'] * df['Units_Sold']
print(df)





# Find the category with the highest total revenue:
highest_revenue_category = df.groupby('Category')['Revenue'].sum().idxmax()
print(highest_revenue_category)




# Plot a bar chart showing Revenue for each product:
import matplotlib.pyplot as plt

df.plot(kind='bar', x='Product', y='Revenue', title='Revenue by Product')
plt.show()






# 
data = {
    "Name": ["Ali", "Sara", "Ahmed", "Zoya", None],
    "Marks": [85, 90, None, 88, 76],
    "City": ["Lahore", "Karachi", None, "Islamabad", "Lahore"]
}
df = pd.DataFrame(data)
print(df)
# Fill missing values in Marks with column mean
df['Marks'].fillna(df['Marks'].mean(), inplace=True)

# Replace missing values in City with 'Unknown'
df['City'].fillna('Unknown', inplace=True)

# Drop rows where Name is missing
df.dropna(subset=['Name'], inplace=True)

print(df)








# 
data = {
    "OrderID": [1, 2, 3, 4, 5],
    "Customer": ["Ali", "Sara", "Ahmed", "Ali", "Zoya"],
    "Product": ["Laptop", "Tablet", "Laptop", "Monitor", "Tablet"],
    "Price": [80000, 30000, 80000, 15000, 30000],
    "Quantity": [1, 2, 1, 1, 3]
}
df = pd.DataFrame(data)

df['Revenue'] = df['Price'] * df['Quantity']
customer_revenue = df.groupby('Customer')['Revenue'].sum()
print(customer_revenue)


# Identify the product with the highest sales (quantity):
highest_sales_product = df.groupby('Product')['Quantity'].sum().idxmax()
print(highest_sales_product)



# Display orders with Price > 50,000:
orders = df[df['Price'] > 50000]
print(orders)






# Merge datasets on StudentID:

students = pd.DataFrame({
    'StudentID': [1, 2, 3],
    'Name': ['Ali', 'Sara', 'Ahmed'],
    'Age': [20, 22, 21]
})

grades = pd.DataFrame({
    'StudentID': [1, 2, 3],
    'Subject': ['Math', 'Science', 'English'],
    'Marks': [85, 78, 88]
})

merged_df = pd.merge(students, grades, on='StudentID')
print(merged_df)
# Calculate the average marks of all students:

avg_marks = merged_df['Marks'].mean()
print(avg_marks)
# Sort the merged DataFrame by Marks in descending order:

sorted_df = merged_df.sort_values(by='Marks', ascending=False)
print(sorted_df)


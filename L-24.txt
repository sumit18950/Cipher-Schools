import numpy as np
import pandas as pd
from tabulate import tabulate

# Generate random data
data = {
    'Age': np.random.normal(30, 10, 100).astype(int),
    'Annual Income': np.random.normal(50, 20, 100).astype(int),
    'Spending Score': np.random.normal(1, 100, 100).astype(int),
    'Credit Score': np.random.normal(600, 100, 100).astype(int),
    'Debt': np.random.normal(1000, 500, 100).astype(int),
    'Credit Limit': np.random.normal(5000, 2000, 100).astype(int)
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate correlation matrix
matrix = df.corr()

# Display correlation matrix using tabulate
print(tabulate(matrix, headers='keys', tablefmt='grid', numalign='right', floatfmt=".2f"))

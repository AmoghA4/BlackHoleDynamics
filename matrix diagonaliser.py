import os
import numpy as np
import pandas as pd
from scipy.linalg import eig

# Parameters
lower_limit = 0
upper_limit = 100
size = 500

# Path to desktop
desktop_path = os.path.expanduser("~/Desktop")

# Random matrix generator
random_matrix = np.random.randint(lower_limit, upper_limit, (size, size))
random_matrix_file_path = os.path.join(desktop_path, "Random Matrix.txt")
np.savetxt(random_matrix_file_path, random_matrix, fmt='%d')

# Defining file paths for eigenvectors and eigenvalues
Eigenvector_file_path = os.path.join(desktop_path, "Eigenvector Matrix.txt")
Eigenvalue_file_path = os.path.join(desktop_path, "Eigenvalue Matrix.txt")

# Perform eigenvalue decomposition 
eigenvalues, eigenvectors = eig(random_matrix)

# Save the eigenvectors and eigenvalues 
np.savetxt(Eigenvector_file_path, np.real(eigenvectors), fmt='%.6f')
np.savetxt(Eigenvalue_file_path, np.diag(np.real(eigenvalues)), fmt='%.6f')

# Create a Pandas Excel writer
excel_file_path = os.path.join(desktop_path, "Matrix.xlsx")
with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    # Convert matrices to DataFrames 
    random_matrix_df = pd.DataFrame(random_matrix)
    eigenvectors_df = pd.DataFrame(np.real(eigenvectors))
    eigenvalues_df = pd.DataFrame(np.diag(np.real(eigenvalues)))

    # Write each matrix to a separate sheet
    random_matrix_df.to_excel(writer, sheet_name='Random Matrix', index=False, header=False)
    eigenvectors_df.to_excel(writer, sheet_name='Eigenvector Matrix', index=False, header=False)
    eigenvalues_df.to_excel(writer, sheet_name='Eigenvalue Matrix', index=False, header=False)

# Print confirmation
print(f"Matrix diagonalization completed.")
print(f"Random matrix saved to: {random_matrix_file_path}")
print(f"Eigenvector matrix saved to: {Eigenvector_file_path}")
print(f"Diagonal matrix saved to: {Eigenvalue_file_path}")
print(f"All matrices saved in Excel file: {excel_file_path}")
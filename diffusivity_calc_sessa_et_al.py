# membrane curvature calculation
# The influence of hydroxylation of sphingolipids on membrane physical state
# Lucia Sessa, Stefano Piotto, Francesco Marrafino, Barbara Panunzi, Rosita Diana, Simona Concilio
# lucsessa@unisa.it

!pip install pandas
!pip install matplotlib numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile

def calculate_mean(file_path):
    df = pd.read_csv(file_path, sep='\t', skiprows=1, header=None, engine='python')
    data = df.iloc[:, 1:]
    data = data.mask(data > 80, np.nan)
    row_means = data.mean(axis=1, skipna=True)
    return row_means, df[0]

def calculate_D(msd_values, time_values):
    D_values = msd_values / (4 * time_values)
    return D_values

def calculate_diff_coeff(last_10_D_values):
    diff_coeff = last_10_D_values.mean()
    stdev_last_10_D = last_10_D_values.std()
    return diff_coeff, stdev_last_10_D

# Directory containing the txt files
directory = '/content/*.txt'

# List to store diffusion coefficients and standard deviations for each file
results_list = []

# Iterate through each txt file in the directory
for file_name in os.listdir(directory):
    if file_name.endswith(".txt"):
        file_path = os.path.join(directory, file_name)

        # Example usage
        msd_values, time_values = calculate_mean(file_path)

        # Extract the base name of the file without the extension
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Calculate D values
        D_values = calculate_D(msd_values, time_values)

        # Plot D against time
        plt.plot(time_values, D_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Time (ns)')
        plt.ylabel('D')
        plt.title(f'D vs Time - {base_file_name}')
        plt.grid(True)

        # Save the plot as an image with a dynamic name based on the input file
        plt.savefig(f'/content/{base_file_name}_D_vs_Time_plot.png', dpi=300)

        # Show the plot
        plt.show()

        # Calculate diffusion coefficient and its standard deviation based on the last 10 values of D
        last_10_D_values = D_values.tail(10)
        diff_coeff, stdev_last_10_D = calculate_diff_coeff(last_10_D_values)

        # Append results to the list
        results_list.append([base_file_name, diff_coeff, stdev_last_10_D])

        # Print diffusion coefficient and its standard deviation
        print(f"\nFile: {base_file_name}")
        print("Diffusion Coefficient (Å^2/ns):", diff_coeff)
        print("Standard Deviation:", stdev_last_10_D)
        print("-" * 50)

# Write results to a single txt file
results_df = pd.DataFrame(results_list, columns=['File Name', 'Diffusion Coefficient (Å^2/ns)', 'Standard Deviation'])
results_df.to_csv('/content/results_summary.txt', sep='\t', index=False)

# Create a zip file containing all txt and png files
with zipfile.ZipFile('/content/results.zip', 'w') as zipf:
    for file_name in os.listdir(directory):
        if file_name.endswith((".txt", ".png")):
            file_path = os.path.join(directory, file_name)
            zipf.write(file_path, os.path.basename(file_path))
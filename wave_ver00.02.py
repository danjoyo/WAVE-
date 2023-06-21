# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:45:44 2023

@author: danjoyo
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

def calculate_average_wave_height(wave_heights):
    average_height = sum(wave_heights) / len(wave_heights)
    return average_height

def calculate_significant_wave_height(wave_heights):
    sorted_wave_heights = sorted(wave_heights, reverse=True)
    one_third_index = math.ceil(len(sorted_wave_heights) / 3)
    highest_third = sorted_wave_heights[:one_third_index]
    significant_height = sum(highest_third) / len(highest_third)
    return significant_height

# Read wave data from Excel file
df = pd.read_excel("wave_data01.xlsx")

# Group data by "Year"
grouped_data = df.groupby("Year")

table_data = []
for year, group in grouped_data:
    heights = group["Height"].tolist()
    average_height = calculate_average_wave_height(heights)
    significant_height = calculate_significant_wave_height(heights)
    max_height = max(heights)
    rms_height = np.sqrt(np.mean(np.square(heights)))  # Calculate RMS
    table_data.append([year, average_height, significant_height, max_height, rms_height])

# Create a DataFrame with the table data
df_table = pd.DataFrame(table_data, columns=["Year", "Hmean (m)", "Hsig (m)", "Hmax (m)", "Hrms (m)"])
df_table = df_table.round(decimals=4)  # Limit decimals to 4 places

# Print the DataFrame
print(df_table)

# Plot the DataFrame as a table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")  # Hide axis
table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()

# Plot wave height vs wave period
wave_heights = df['Height']
periods = df['Period']

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = linregress(periods, wave_heights)
linear_trend_line = intercept + slope * periods

# Calculate exponential regression
exp_regression_coeffs = np.polyfit(periods, np.log(wave_heights), 1)
exp_trend_line = np.exp(exp_regression_coeffs[1]) * np.exp(exp_regression_coeffs[0] * periods)

# Plot the data and trend lines
plt.figure(figsize=(16, 12))
plt.scatter(periods, wave_heights, label="Data")
plt.plot(periods, linear_trend_line, color="red", label="Linear Trend Line")
plt.plot(periods, exp_trend_line, color="green", label="Exponential Trend Line")
plt.xlabel("Period (detik)")
plt.ylabel("Tinggi Gelombang (m)")
plt.title("Tinggi Gelombang vs. Periode")
plt.legend()
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.xticks(np.arange(0, max(periods)+1, 0.5))
plt.yticks(np.arange(0, max(wave_heights)+1, 0.2))

# Add equation functions to the plot
equation_linear = f"Linear: y = {slope:.2f}x + {intercept:.2f}"
equation_exponential = f"Exponential: y = {np.exp(exp_regression_coeffs[1]):.2f} * e^({exp_regression_coeffs[0]:.2f}x)"
plt.text(max(periods), max(wave_heights), equation_linear, ha='right', va='bottom', color='red')
plt.text(max(periods), max(wave_heights) - 0.2, equation_exponential, ha='right', va='bottom', color='green')

plt.show()

import matplotlib.pyplot as plt

# Data for years and corresponding deepfake numbers
years = [2020, 2021, 2022, 2023, 2024, 2025]
deepfakes = [7813, 31250, 125000, 500000, 2000000, 8000000]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(years, deepfakes, marker='o', linestyle='-')

# Add title and labels
plt.title('Rise of Deepfakes Shared on Social Media (2020-2025)')
plt.xlabel('Year')
plt.ylabel('Number of Deepfakes Shared')

# Add grid for better readability
plt.grid(True)

# Save the plot to a file
plt.savefig('deepfake_growth.png')

import numpy as np
import matplotlib.pyplot as plt
noisy_matrix = np.load('cost_matrices_output/noisy.npy')

plt.figure(figsize=(8, 6))
plt.imshow(noisy_matrix, cmap='viridis')
plt.colorbar(label='Cost')
plt.title('Similarity Matrix of Noisy Data')
plt.xlabel('Point index')
plt.ylabel('Point index')

output_path = 'cost_matrix_visualizations/noisy_cost_matrix_heatmap.png'
plt.savefig(output_path)

print(f"ok : {output_path}")

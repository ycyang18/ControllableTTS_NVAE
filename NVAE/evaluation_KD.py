
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

#kernal='gaussian'
kde_generated = KernelDensity(bandwidth=0.1).fit(generated_embeddings)
kde_truth = KernelDensity(bandwidth=0.1).fit(distribution_data)

x = np.linspace(min(generated_embeddings.min(), distribution_data.min()), max(generated_embeddings.max(), distribution_data.max()), 1000)
pdf_generated = np.exp(kde_generated.score_samples(x.reshape(-1, 1)))
pdf_truth = np.exp(kde_truth.score_samples(x.reshape(-1, 1)))

plt.plot(x, density_sample, label='Generated Embeddings KDE')
plt.plot(x, pdf_truth, label='True Embeddings KDE')
plt.legend()
plt.show()

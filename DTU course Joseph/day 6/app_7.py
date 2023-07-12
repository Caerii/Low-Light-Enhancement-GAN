import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import MiniBatchDictionaryLearning

# Load the Barbara image
barbara = io.imread('barbara.png', as_gray=True)

# Create a random mask with 50% missing pixels
mask = np.random.rand(*barbara.shape) < 0.5

# Apply the mask to the image
barbara_masked = np.multiply(barbara, mask)

# Define the GMCA parameters
Ns = 3
Nc = 1
n_iter = 500
n_atoms = 64
alpha = 1

# Reshape the data for GMCA
X = barbara_masked.reshape(-1,1)

# Initialize the dictionary for GMCA
dico = MiniBatchDictionaryLearning(n_components=n_atoms, alpha=alpha, n_iter=50)
D = dico.fit(X).components_

# Apply GMCA to the masked image
A = np.random.rand(Nc, Ns)
for i in range(n_iter):
    S = np.dot(np.linalg.pinv(A), X)
    for j in range(Ns):
        Aj = A[:,j].reshape(-1,1)
        Sj = S[j,:].reshape(-1,1)
        xj = np.dot(Aj, Sj).reshape(-1,1)
        mj = np.multiply(mask, xj)
        Sj = dico.transform(mj.T)
        A[:,j] = np.squeeze(dico.components_.T)

# Reconstruct the image from the estimated sources
barbara_inpaint = np.dot(A, S).reshape(barbara.shape)

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
ax = axes.ravel()
ax[0].imshow(barbara, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(barbara_masked, cmap=plt.cm.gray)
ax[1].set_title('Masked')
ax[2].imshow(barbara_inpaint, cmap=plt.cm.gray)
ax[2].set_title('Inpainted')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()
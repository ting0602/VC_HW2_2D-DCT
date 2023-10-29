import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 2D-DCT
def dct2_manual(img):
    M, N = img.shape
    dct_result = np.zeros((M, N), dtype=np.float64)
    
    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    dct_result[u, v] += img[x, y] * np.cos((2*x + 1)*u*np.pi / (2*M)) * np.cos((2*y + 1)*v*np.pi / (2*N))
            dct_result[u, v] *= 2/N
            if u == 0:
                dct_result[u, v] /= np.sqrt(2)
            if v == 0:
                dct_result[u, v] /= np.sqrt(2)
    return dct_result

# 2D-IDCT
def idct2_manual(dct_coeffs):
    M, N = dct_coeffs.shape
    img_result = np.zeros((M, N), dtype=np.float64)
    
    for x in range(M):
        for y in range(N):
            for u in range(M):
                for v in range(N):
                    cu = 1/np.sqrt(2) if u == 0 else 1
                    cv = 1/np.sqrt(2) if v == 0 else 1
                    img_result[x, y] += cu * cv * dct_coeffs[u, v] * np.cos((2*x + 1)*u*np.pi / (2*M)) * np.cos((2*y + 1)*v*np.pi / (2*N))
            img_result[x, y] *= 2/N
    return img_result


# Visualize coefficients in the log domain
def visualize_dct(dct_coeffs):
    # Take the log of absolute values for visualization
    log_coeffs = np.log(np.abs(dct_coeffs))
    
    # Normalize for visualization
    log_coeffs -= np.min(log_coeffs)
    log_coeffs /= np.max(log_coeffs)
    
    return log_coeffs

# 1D DCT
def dct1(x):
    N = len(x)
    X = np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * k * (2*n+1) / (2*N))
        X[k] *= np.sqrt(2/N)
        if k == 0:
            X[k] /= np.sqrt(2)
    return X

# two 1D DCT
def dct1_manual(img):
    M, N = img.shape
    dct_result = np.zeros((M, N), dtype=np.float64)
    
    # Apply 1D DCT to each row
    for a in range(M):
        dct_result[a, :] = dct1(img[a, :])
        
    # Apply 1D DCT to each column
    for b in range(N):
        dct_result[:, b] = dct1(dct_result[:, b])
        
    return dct_result

# Evaluate the PSNR
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel_value = np.max(original)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

# load image
lena = cv2.imread("lena.png")
lena = cv2.resize(lena, (64, 64), interpolation=cv2.INTER_AREA)
# Convert the image to grayscale
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
lena = lena.astype(np.float64)

# Compute 2D-DCT
start_time_2d = time.time()
lena_dct_manual = dct2_manual(lena/255.0)
end_time_2d = time.time()
print(f'Time taken for 2D-DCT: {end_time_2d - start_time_2d} seconds')
visualized_dct = visualize_dct(lena_dct_manual)
plt.figure(figsize=(4, 4))
plt.imshow(visualized_dct, cmap='gray', extent=(0, visualized_dct.shape[1], 0, visualized_dct.shape[0]))
plt.title('2D-DCT Coefficients (log domain)')
plt.colorbar()
plt.savefig('2D-DCT_coefficients.png')
plt.close()

# Reconstruct the image using 2D-IDCT
lena_reconstructed = idct2_manual(lena_dct_manual)
lena_reconstructed = (lena_reconstructed * 255)
lena_reconstructed = idct2_manual(lena_dct_manual)

# Clip values
lena_reconstructed = ((lena_reconstructed - np.min(lena_reconstructed)) / (np.max(lena_reconstructed) - np.min(lena_reconstructed)))

plt.figure(figsize=(4, 4))
plt.imshow(lena_reconstructed, cmap='gray', extent=(0, lena_reconstructed.shape[1], 0, lena_reconstructed.shape[0]))
plt.title('Reconstructed Image')
plt.savefig('2D-IDCT_reconstructed.png')
plt.close()

# two 1D DCT
start_time_1d = time.time()
lena_1dct_manual = dct1_manual(lena/255.0)
end_time_1d = time.time()
print(f'Time taken for two 1D-DCT: {end_time_1d - start_time_1d} seconds')

visualized_1dct = visualize_dct(lena_1dct_manual)
plt.figure(figsize=(4, 4))
plt.imshow(visualized_1dct, cmap='gray', extent=(0, visualized_1dct.shape[1], 0, visualized_1dct.shape[0]))
plt.title('two 1D-DCT Coefficients (log domain)')
plt.colorbar()
plt.savefig('1D-DCT_coefficients.png')
plt.close()

# calculate PSNR
psnr = calculate_psnr(lena, lena_reconstructed)
print(f'PSNR: {psnr} dB')

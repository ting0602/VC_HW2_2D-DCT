# VC_HW2_2D-DCT

**NYCU Video Compression HW2**

- **2D-DCT**
  - Implement 2D-DCT to transform “lena.png” to DCT coefficients (visualize in log domain).
    - Convert the input image to grayscale first.
    - Visualize the coefficients in the log domain. Feel free to scale and clip the coefficients for visualization.
  - Implement 2D-IDCT to reconstruct the image.
  - Evaluate the PSNR.

- **Two 1D-DCT**
  - Implement a fast algorithm by two 1D-DCT to transform “lena.png ” to DCT coefficients.

- Compare the runtime between 2D-DCT and two 1D-DCT.
- Do not use any functions for DCT and IDCT, e.g., cv2.dct

## Project Structure
The project includes the following files:

```109550119.py``` - The main Python script that contains the manual implementation of 2D-DCT, PSNR calculation, and image compression and reconstruction.

```lena.png``` - The input image file, using the Lena image in this example.

```2D-DCT_coefficients.png``` - An image containing 2D-DCT coefficients for visualization.

```2D-IDCT_reconstructed.png``` - The image reconstructed using 2D-DCT, available for visualization.

```1D-DCT_coefficients.png``` - Images containing coefficients for two 1D-DCT operations for comparison.

Hint: Since the original image size is too large, I first produced the image to 64x64 for processing.

## Dependencies
To run this project, you need to have the following libraries installed:

- Python
- NumPy
- OpenCV
- Matplotlib

## Usage
### Clone Repository
```
git clone https://github.com/ting0602/VC_HW2_2D-DCT.git
```

### Run the Code
```
python 109550119.py
```


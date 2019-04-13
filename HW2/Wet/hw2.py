import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from scipy import signal
from pyunlocbox import functions
from pyunlocbox import solvers
import casm

KERNEL_SIZE = 12
ALPHA = 2.5
sigma = 5

# 2D-Gaussian kernel
def gaussian_psf(kernel_size, sigma_param):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma_param**2))
    return kernel / np.sum(kernel)


# 2D-Box kernel
def box_psf(kernel_size):
    box = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            box[i][j] = 1/(kernel_size**2)
    return box

# gets the low resolution image and k creates and returns the high resolution kernel using Wiener filter.
def l_to_h_wf(l_org, k):
    l = l_org
    L_fft = np.fft.fft2(l)
    k_padded = np.pad(k, [((l.shape[0] - k.shape[0])//2,), ((l.shape[1] - k.shape[1])//2,)], 'constant')
    k_padded = np.pad(k_padded, (0, 1), mode='constant')
    K_fft = np.fft.fft2(k_padded)
    W_filter = np.conj(K_fft)/(np.linalg.norm(K_fft, 1) ** 2)
    H_fft = W_filter * L_fft
    return np.abs(np.fft.fftshift(np.fft.ifft2(H_fft)))


if __name__ == '__main__':

    # Task 1
    # High-res gaussian PSF
    gaussian_H = gaussian_psf(KERNEL_SIZE, sigma)
    # Low-res gaussian PSF
    gaussian_L = gaussian_psf(int(KERNEL_SIZE * ALPHA), sigma)

    plt.figure(1)
    plt.imshow(gaussian_H, cmap='gray')
    plt.title("Gaussian PSF_H kernel")

    plt.figure(2)
    plt.imshow(gaussian_L, cmap='gray')
    plt.title("Gaussian PSF_L kernel")

    # Task 2
    # High-res box function PSF
    box_H = box_psf(KERNEL_SIZE)
    # Low-res box function PSF
    box_L = box_psf(int(KERNEL_SIZE*ALPHA))

    plt.figure(3)
    plt.imshow(box_H, cmap='gray')
    plt.title("Box PSF_H kernel")

    plt.figure(4)
    plt.imshow(box_L, cmap='gray')
    plt.title("Box PSF_L kernel")

    # Task 3
    original_image = cv2.imread('DIPSourceHW2.png', cv2.IMREAD_GRAYSCALE).astype(float)

    plt.figure(5)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original image")

    blurred_image_gaussian_h = signal.convolve2d(original_image, gaussian_H, boundary='symm', mode='same')
    plt.figure(6)
    plt.imshow(blurred_image_gaussian_h, cmap='gray')
    plt.title("Blurred with high-res gaussian PSF")

    blurred_image_gaussian_l = signal.convolve2d(original_image, gaussian_L, boundary='symm', mode='same')
    plt.figure(7)
    plt.imshow(blurred_image_gaussian_l, cmap='gray')
    plt.title("Blurred with low-res gaussian PSF")

    blurred_image_box_h = signal.convolve2d(original_image, box_H, boundary='symm', mode='same')
    plt.figure(8)
    plt.imshow(blurred_image_box_h, cmap='gray')
    plt.title("Blurred with high-res box PSF")

    blurred_image_box_l = signal.convolve2d(original_image, box_L, boundary='symm', mode='same')
    plt.figure(9)
    plt.imshow(blurred_image_box_l, cmap='gray')
    plt.title("Blurred with low-res box PSF")

    # Task 4
    # We concluded in the dry part that:
    # H*K = L
    # Thus, we need to find the convolution toplitz matrix.
    # multiply it with the vector k which (create from the kernel).

    doubly_blocked_h = casm.convert_to_toeplitz_doubely_blocked((19, 19), gaussian_H)
    k = np.linalg.lstsq(doubly_blocked_h, casm.matrix_to_vector(gaussian_L), rcond=None)
    K_gaussian = casm.vector_to_matrix(k[0], (19, 19))

    plt.figure(10)
    plt.title("Gaussian blur kernel")
    plt.imshow(K_gaussian, cmap='gray')

    doubly_blocked_h = casm.convert_to_toeplitz_doubely_blocked((19, 19), box_H)
    k = np.linalg.lstsq(doubly_blocked_h, casm.matrix_to_vector(box_L), rcond=None)
    K_box = casm.vector_to_matrix(k[0], (19, 19))

    plt.figure(11)
    plt.title("Box blur kernel")
    plt.imshow(K_box, cmap='gray')

    recon_gauss = signal.convolve2d(gaussian_H, K_gaussian)
    recon_box = signal.convolve2d(box_H, K_box)

    plt.figure(12)
    plt.title("Reconstructed gaussian PSF_L")
    plt.imshow(recon_gauss, cmap='gray')
    plt.figure(13)
    plt.title("Reconstructed box PSF_L")
    plt.imshow(recon_box, cmap='gray')

    # Task 5.1

    H_estimated_wf_box = l_to_h_wf(blurred_image_box_l, K_box)
    H_estimated_wf_gaussian = l_to_h_wf(blurred_image_gaussian_l, K_gaussian)

    plt.figure(14)
    plt.title("Estimated Super resolution gaussian - Wiener filter")
    plt.imshow(H_estimated_wf_gaussian, cmap='gray')
    plt.figure(15)
    plt.title("Estimated Super resolution box - Wiener filter")
    plt.imshow(H_estimated_wf_box, cmap='gray')

    # Task 5.2
    tau = 100

    g = lambda H:  signal.convolve2d(H, K_gaussian, boundary='symm', mode='same')
    l_blurred_cpy = np.array(blurred_image_gaussian_l)
    tv_prior_f = functions.norm_tv(maxit=50, dim=2)
    norm_l2_f = functions.norm_l2(y=l_blurred_cpy, A=g, lambda_=tau)
    solver = solvers.forward_backward(step=0.0001 / tau)
    H_estimated_lms_tv_gaussian = solvers.solve([tv_prior_f, norm_l2_f], l_blurred_cpy, solver, maxit=100)

    g = lambda H:  signal.convolve2d(H, K_box, boundary='symm', mode='same')
    l_blurred_cpy = np.array(blurred_image_box_l)
    tv_prior_f = functions.norm_tv(maxit=50, dim=2)
    norm_l2_f = functions.norm_l2(y=l_blurred_cpy, A=g, lambda_=tau)
    solver = solvers.forward_backward(step=0.0001 / tau)
    H_estimated_lms_tv_box = solvers.solve([tv_prior_f, norm_l2_f], l_blurred_cpy, solver, maxit=100)

    plt.figure(16)
    plt.title("Estimated Super resolution gaussian - Least mean square with TV prior")
    plt.imshow(H_estimated_lms_tv_gaussian['sol'], cmap='gray')

    plt.figure(17)
    plt.title("Estimated Super resolution box - Least mean square with TV prior")
    plt.imshow(H_estimated_lms_tv_box['sol'], cmap='gray')

    # Task 6
    # Bilinear filter over gaussian psf
    bilinear_gaussian = skimage.transform.resize(blurred_image_gaussian_l, (blurred_image_gaussian_l.shape[0]*ALPHA, blurred_image_gaussian_l.shape[1]*ALPHA))
    plt.figure(18)
    plt.title("Bilinear upsampling over gaussian psf")
    plt.imshow(bilinear_gaussian, cmap='gray')

    # Bicubic filter over gaussian psf
    bicubic_gaussian = skimage.transform.resize(blurred_image_gaussian_l, (blurred_image_gaussian_l.shape[0]*ALPHA, blurred_image_gaussian_l.shape[1]*ALPHA), order=3)
    plt.figure(19)
    plt.title("Bicubic upsampling over gaussian psf")
    plt.imshow(bicubic_gaussian, cmap='gray')

    # Bilinear filter over box psf
    bilinear_box = skimage.transform.resize(blurred_image_box_l, (blurred_image_gaussian_l.shape[0]*ALPHA, blurred_image_gaussian_l.shape[1]*ALPHA))
    plt.figure(20)
    plt.title("Bilinear upsampling over box psf")
    plt.imshow(bilinear_box, cmap='gray')

    # Bicubic filter over box psf
    bicubic_box = skimage.transform.resize(blurred_image_box_l, (blurred_image_gaussian_l.shape[0]*ALPHA, blurred_image_gaussian_l.shape[1]*ALPHA), order=3)
    plt.figure(21)
    plt.title("Bicubic upsampling over box psf")
    plt.imshow(bicubic_box, cmap='gray')

    plt.show()


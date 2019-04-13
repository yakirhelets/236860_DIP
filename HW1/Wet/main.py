from matplotlib import pyplot as plt
import scipy.io as sio
import scipy.misc
from scipy import signal, fftpack
from scipy.ndimage import gaussian_filter
import numpy as np
import imageio
import cv2
import os

KERNEL_SIZE = 17

# PART I
def trajectory_and_psf_blurring():

    # Part 1.1
    mat = sio.loadmat('100_motion_paths.mat')
    data_x = mat['X']  # X values for each trajectory
    data_y = mat['Y']  # Y values for each trajectory

    # Plotting the trajectories, each one is a subplot in 10 X 10 grid
    plt.figure(1)
    for k in range(len(data_x)):
        plt.subplot(10, 10, k+1)
        plt.plot(data_x[k], data_y[k])

    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Trajectories resulted by the handshake")
    plt.show()

    # Part 1.2
    psf_kernel_list = []
    max_value = 0
    center = KERNEL_SIZE // 2

    if os.path.isdir('./output_psf'):
        for file in os.listdir('./output_psf'):
            os.remove(os.path.join('./output_psf', file))
    else:
        os.mkdir('./output_psf')

    # Creating 100 psf kernels from each trajectory
    # The creation process of each psf kernel:
    # For each [x,y] in the trajectory vector add 1 to the [center - round(y),center - round(x)] corresponding cell.
    # in order to get a valid value range divide every psf kernel with the maximum value found in overall cells.
    for idx in range(len(data_x)):
        psf_kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
        for k in range(len(data_x[idx])):
            psf_kernel[center - int((np.sign(data_y[idx, k])) * round(abs(data_y[idx, k]))),
                       center + int((np.sign(data_x[idx, k])) * round(abs(data_x[idx, k])))] += 1
        psf_kernel_list.append(psf_kernel)

        if psf_kernel.max() > max_value:
            max_value = psf_kernel.max()

    for idx in range(len(data_x)):
        psf_kernel_list[idx] /= max_value
        psf_kernel_list[idx] *= 255
        psf_kernel_list[idx] = psf_kernel_list[idx].astype(np.uint8)
        # Saving the psf kernel as image.
        imageio.imwrite('./output_psf/psf' + ("0" + str(idx) if idx < 10 else str(idx)) + '.png', psf_kernel_list[idx])

    # Plotting the psf kernels, each one is a subplot in 10 X 10 grid
    plt.figure(2)
    for idx in range(len(data_x)):
        plt.subplot(10, 10, idx+1)
        plt.imshow(psf_kernel_list[idx], cmap='gray')  # PSF matrix created
    plt.show()

    # Part 1.3

    if os.path.isdir('./output_blurred'):
        for file in os.listdir('./output_blurred'):
            os.remove(os.path.join('./output_blurred', file))
    else:
        os.mkdir('./output_blurred')

    # Generating blurred images from an original given one, by doing a convolution with each kernel.
    original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
    for idx in range(len(data_x)):
        blurred_image = signal.convolve2d(original_image, psf_kernel_list[idx]/255.0, boundary='symm', mode='same').clip(0, 255)
        # Saving the blurred image.
        imageio.imwrite('./output_blurred/blurred' + ("0"+str(idx) if idx < 10 else str(idx)) + '.png', blurred_image.astype(np.uint8))

# PART 2
def deblurring():
    # Loading the blurred images from part 1.
    image_list = os.listdir('./output_blurred')
    fft_image_list = []
    fft_abs_list = []
    # Making lists of the fourier transformed images and their absolute value.
    for image in image_list:
        curr_image = cv2.imread(os.path.join('./output_blurred', image), cv2.IMREAD_GRAYSCALE).astype(float)/255.0
        fft_image_list.append(fftpack.fft2(curr_image))
        fft_abs_list.append(fftpack.fftshift(abs(fftpack.fft2(curr_image))))

    # Applying low pass filter to fourier transformed absolute value list using a gaussian filter.
    sigma = 256 / float(KERNEL_SIZE)
    gaussian_abs_list = [fftpack.ifftshift(gaussian_filter(x, sigma)) for x in fft_abs_list]
    p = 11

    # Loading original image for PSNR computation .
    original_image = cv2.imread('DIPSourceHW1.jpg', cv2.IMREAD_GRAYSCALE).astype(float)

    if os.path.isdir('./output_deblurred'):
        for file in os.listdir('./output_deblurred'):
            os.remove(os.path.join('./output_deblurred', file))
    else:
        os.mkdir('./output_deblurred')

    psnr_list = []
    # Generating deblurred images for changing number of frames (changing k first frames) by this formula:
    # inverse_fourier(sum(image_fourier_transform_i*weight_i/total_weight))
    # weight_i = (|image_fourier_transform_i|*Gaussian_sigma)^p
    for num_of_images in range(1, len(image_list)+1):
        # calc the weights out of the photos
        total_weight = np.zeros((256, 256)).astype(float)
        fft_weight_list = []
        for idx in range(num_of_images):
            total_weight += gaussian_abs_list[idx] ** p
        for idx in range(num_of_images):
            fft_weight_list.append(gaussian_abs_list[idx] ** p / total_weight)

        final_image = np.zeros((256, 256)).astype(complex)
        for idx in range(num_of_images):
            final_image += (fft_image_list[idx] * fft_weight_list[idx])

        final_image = fftpack.ifft2(final_image).real * 255.0

        # Saving the first k-th frames deblurred image.
        imageio.imwrite('./output_deblurred/deblurred' + ("0" + str(num_of_images-1) if (num_of_images-1) < 10 else str(num_of_images-1)) + '.png', final_image.astype(np.uint8))

        # PSNR calculation using the following formula:
        # 10*np.log10(MAX**2/MSE).
        # MAX - max pixel value of the original image.
        # MSE - its just the MSE between the original and the deblurred images.
        MSE = ((original_image - final_image)**2).mean()
        MAX = original_image.max()

        psnr_list.append(10*np.log10(MAX**2/MSE))

    # Plotting graph of the PSNR values as function of number of images
    plt.figure(3)
    plt.plot(psnr_list)

    plt.xlabel("Number of images")
    plt.ylabel("PSNR value")
    plt.title("PSNR values as function of number of images")
    plt.show()

def main():
    trajectory_and_psf_blurring()
    deblurring()

if __name__ == "__main__":
    main()
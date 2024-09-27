# importing necessary libraries
import ee
import numpy as np
import pylab as plt
import cv2
from numpy.fft import rfftn, irfftn, fftshift
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from roboflow import Roboflow
import os
###############################################################
API_KEY = ""

###############################################################
rf = Roboflow(api_key=API_KEY)
project = rf.workspace('fypwavemapping').project('cloud-detection-nsk0q').version(1)
model = project.model
print('done')

###############################################################
def get_img_coordinate(dataset, coordinate, offset):
    # obtaining rectangle region
    lat = coordinate[1]
    lon = coordinate[0]
    rectangle = ee.Geometry.Rectangle(lon-offset, lat-offset, lon+offset, lat+offset)
    xy = ee.Geometry.Point([lat, lon])

    dataset = dataset.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    dataset_region = dataset.filterBounds(rectangle)

    sample_img = dataset_region.first()
    bands_master = sample_img.sampleRectangle(rectangle)

    return np.array(bands_master.get("B4").getInfo(), dtype=np.uint64)

def find_clouds(img):
    # temporary saving the image
    img_path = 'pred_img.jpg'
    plt.imsave(img_path, img)

    # predict clouds
    preds = model.predict(img_path).predictions

    if len(preds) > 0:
        preds = preds.pop()
    else:
        return

    # removing temporary image
    os.remove(img_path)

    return preds

def cloud_exclusion(img):
    preds = find_clouds(img)

    # creating cloud mask
    if preds is not None:
        mask = np.full((img.shape[0], img.shape[1]), np.nan)
        pts = np.array([(int(point['x']), int(point['y'])) for point in preds['points']], dtype=np.int32)

        # draw the polygon on the mask
        cv2.fillPoly(mask, [pts], color=255)
        
        cloudless = np.where(np.isnan(mask), img, np.nan)
        cloudless = np.where(np.isnan(cloudless), np.median(cloudless[~np.isnan(cloudless)]), img)

        return cloudless, mask
    else:
        return None, None
    
###############################################################

def fourier_high_pass(img, r):
    f_trans = np.fft.fftshift(np.fft.fft2(img))

    c_y, c_x = img.shape[0]//2, img.shape[1]//2

    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = np.sqrt((x - c_x)**2 + (y - c_y)**2) <= r

    f_trans[mask] = 0

    f_trans_unshift = np.fft.ifftshift(f_trans)
    img_filtered = np.fft.ifft2(f_trans_unshift).real

    return img_filtered

def fourier_low_pass(img, r):
    f_trans = np.fft.fftshift(np.fft.fft2(img))

    c_y, c_x = img.shape[0]//2, img.shape[1]//2

    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = np.sqrt((x - c_x)**2 + (y - c_y)**2) <= r

    f_trans[~mask] = 0

    f_trans_unshift = np.fft.ifftshift(f_trans)
    img_filtered = np.fft.ifft2(f_trans_unshift).real
    
    return img_filtered

def z_score_normalize(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    normalized_data = 1 / (1 + np.exp(-z_scores))
    return normalized_data

def sobel(input, sobel_k, normalize_magnitude=True, normalize_angles=True):
    grad_x = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=sobel_k)
    grad_y = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=sobel_k)

    grad_mag = np.sqrt(np.power(grad_y, 2) + np.power(grad_x, 2))
    grad_or = np.arctan2(grad_y, grad_x) * 180/np.pi

    if normalize_magnitude:
        grad_mag = z_score_normalize(grad_mag)

    # quantizing gradient orientation angles
    if normalize_angles:
        grad_or = np.where(grad_or < 0, grad_or+180, grad_or)
        grad_or = np.where(np.abs(grad_or) > 168.75, 0, grad_or)

    return grad_mag, grad_or

def compute_map(img):
    # low-pass filtering
    img = fourier_low_pass(img, 18) #12

    # sobel filtering
    grad_mag, grad_or = sobel(img, 3, True, True)
    
    # creating the output orientation image
    out_waves = np.where(grad_mag>np.mean(grad_mag)+np.std(grad_mag), grad_or, np.nan)
    out_waves = np.array(out_waves, dtype=np.float64)

    return out_waves

def pdf_analysis(waves_out, prominence=0.001, show=False):
    waves_out_flat = waves_out.flatten()
    waves_out_flat = waves_out_flat[~np.isnan(waves_out_flat)]
    kde = gaussian_kde(waves_out_flat)

    x = np.linspace(0, 180, 1000)

    peaks, properties = find_peaks(kde(x), prominence=prominence)
    modes = x[peaks]

    if show:
        plt.plot(x, kde(x))
        plt.title('PDF')
        plt.xlabel('data values')
        plt.ylabel('density')
        plt.show()
    
    return kde, modes

def show_direction(img, waves_out, dir):
    plt.imshow(img)
    plt.imshow(waves_out)
    plt.quiver(img.shape[1]//2, img.shape[0]//2, np.cos(dir * np.pi/180), -np.sin(dir * np.pi/180), scale=3, color='red', width=0.01)
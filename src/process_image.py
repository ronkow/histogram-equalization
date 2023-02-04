import glob
import numpy as np

from pathlib import Path
from pprint import pprint

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2 # for HSV color space, CLAHE

IMAGE_INPUT_DIR = '../data/input/'
IMAGE_OUTPUT_DIR = '../data/output/'

max_intensity = 256

def create_image_dict():
    """
    Create a dictionary {image name:image}
    """
    image_dict = {}
    exts = ['jpg', 'jpeg']

    files =  [glob.glob(IMAGE_INPUT_DIR + f'/*.{ext}') for ext in exts]

    for i in files:
        for f in i:
            fname = Path(f).stem
            image_dict[fname] = mpimg.imread(f)
    return image_dict
    
    
def plot_image_hist(image, image_name, histogram_name):
    """
    Plot image and histogram.
    Save as a PNG file in the directory ./output
    """
    figure, axes = plt.subplots(1,2)
    figure.set_size_inches((12,4)) # width, height of figure
    i, h = axes
    
    i.imshow(image)
    i.set_axis_off()
    i.set_title(image_name)
    
    color = ('r','g','b')
    channel = (0, 1, 2)
    for ch, c in zip(channel, color):    
        image_color = image[:,:,ch]
        h.hist(image_color.flatten(), bins=max_intensity, color=c, alpha=0.6)
        #hist, bin_edges = np.histogram(image_color.flatten(), bins=max_intensity)
        #h.plot(bin_edges[0:-1], hist, color=c)

    h.set_title(histogram_name)
    h.set_xlabel('INTENSITY')
    h.set_ylabel('FREQUENCY')
    h.set_xlim([-5,260])      # white space before 0 and after 250  

    # Save to file
    plt.savefig(IMAGE_OUTPUT_DIR + image_name)    
    
    
def print_mapping(image, mapped_image, method):
    """
    Print mapping {original value:mapped value} for 256 intensity values
    """
    if str(type(image)) == "<class 'list'>": 
        image = np.asarray(image).astype('uint8')

    if str(type(mapped_image)) == "<class 'list'>":
        mapped_image = np.asarray(mapped_image).astype('uint8')
        
    image_1d = image.flatten()
    image_list = image_1d.tolist()
        
    mapped_image_1d = mapped_image.flatten()
    mapped_image_list = mapped_image_1d.tolist()
    
    mapping_dict = {}
    for a, b in zip(image_list, mapped_image_list):
        mapping_dict[a] = b
        
    print(method)
    pprint(mapping_dict)


def map_image(image):      
    """
    Histogram Equalization implementation: Compute mapping of intensity values 0 to 255
    """
    if str(type(image)) == "<class 'list'>":
        image = np.asarray(image).astype('uint8')
 
    image_1d = image.flatten()
    image_list = image_1d.tolist()
 
    N = len(image_list) # Total number of pixels = length of image_1d
    I = max_intensity # 256
        
    ifreq = {}
    imap = {}
    for k in range(0,I):  # Initialize dict with zeros
        ifreq[k] = 0
        imap[k] = 0
        
    for i in image_list:
        ifreq[i] += 1
        
    # For each intensity value from 0 to 255,
    # map it to equalized value
    for k in imap.keys():
        sum = 0
        for i in range(0,k+1): # 0, 1, 2, ..., k
            sum += ifreq[i]
        map_value = ((sum)/(N)) * (I-1)
        imap[k] = round(map_value)
        
    mapped_image_list = []
    for i in image_list:
        i = imap[i]
        mapped_image_list.append(i)    
    return mapped_image_list


def equalize_hist(image):    
    """
    Passes image to map_image() for four different methods of equalization.
    Returns the mapped image.
    Passes image and mapped image to print_mapping().
    """
    method = ''
    
    # SPLIT RGB INTO SEPARATE CHANNELS
    image_list = image.tolist()
    image_r = []
    image_g = []
    image_b = []
    for row in image_list:
        for rgb in row:
            image_r.append(rgb[0])
            image_g.append(rgb[1])
            image_b.append(rgb[2])
     
     
    # METHOD 1. RGB CHANNELS COMBINED
    method = 'RGB COMBINED'
    mapped_image_combined = map_image(image)
    
    # Convert to numpy array and reshape
    mapped_image_combined = np.asarray(mapped_image_combined).astype('uint8')
    mapped_image_combined = np.reshape(mapped_image_combined, image.shape)

    mapped_image_combined_list = mapped_image_combined.tolist()
    mapped_image_combined_r = []
    mapped_image_combined_g = []
    mapped_image_combined_b = []
    for row in mapped_image_combined_list:
        for rgb in row:
            mapped_image_combined_r.append(rgb[0])
            mapped_image_combined_g.append(rgb[1])
            mapped_image_combined_b.append(rgb[2])

    #print_mapping(image_r, mapped_image_combined_r, method + ':R')
    #print_mapping(image_g, mapped_image_combined_g, method + ':G')
    #print_mapping(image_b, mapped_image_combined_b, method + ':B')

    
    # METHOD 2. RGB CHANNELS SEPARATE
    method = 'RGB SEPARATE'        
    mapped_image_r = map_image(image_r)
    mapped_image_g = map_image(image_g)
    mapped_image_b = map_image(image_b)
        
    mapped_rgb = []
    mapped_image_rgb = []
    for r,g,b in zip(mapped_image_r, mapped_image_g, mapped_image_b):
        mapped_rgb = [r,g,b]
        mapped_image_rgb.append(mapped_rgb)       
    
    # Convert to numpy array and reshape
    mapped_image_rgb = np.asarray(mapped_image_rgb).astype('uint8')
    mapped_image_rgb = np.reshape(mapped_image_rgb, image.shape)

    #print_mapping(image_r, mapped_image_r, method + ':R')
    #print_mapping(image_g, mapped_image_g, method + ':G')
    #print_mapping(image_b, mapped_image_b, method + ':B')
    
    # METHOD 3. USING HSV COLOR SPACE
    # Equalize only the V channel
    method = 'HSV'
    image_height, image_width, rgb = image.shape
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channel_hsv = cv2.split(image_hsv)
    
    channel_hsv[2] = map_image(channel_hsv[2].flatten())
    channel_hsv[2] = np.asarray(channel_hsv[2]).astype('uint8')
    channel_hsv[2] = np.reshape(channel_hsv[2], (image_height, image_width))
        
    image_hsv = cv2.merge(channel_hsv)
    mapped_image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    mapped_image_hsv_list = mapped_image_hsv.tolist()
    mapped_image_hsv_r = []
    mapped_image_hsv_g = []
    mapped_image_hsv_b = []
    for row in mapped_image_hsv_list:
        for rgb in row:
            mapped_image_hsv_r.append(rgb[0])
            mapped_image_hsv_g.append(rgb[1])
            mapped_image_hsv_b.append(rgb[2])

    #print_mapping(image_r, mapped_image_hsv_r, method + ':R')
    #print_mapping(image_g, mapped_image_hsv_g, method + ':G')
    #print_mapping(image_b, mapped_image_hsv_b, method + ':B')
    
    # METHOD 4. ADAPTIVE HE (CLAHE) with HSV (4 x 4 GRID)
    method = 'CLAHE WITH HSV'

    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    channel_hsv = cv2.split(image_hsv)  

    # tileGridSize defines the number of tiles in row and column
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    # Do the mapping using cv2. Note that array is not flattened.    
    channel_hsv[2] = clahe.apply(channel_hsv[2])  

    image_hsv_clahe = cv2.merge(channel_hsv)
    mapped_image_hsv_clahe = cv2.cvtColor(image_hsv_clahe, cv2.COLOR_HSV2RGB)

    mapped_image_hsv_clahe_list = []
    mapped_image_hsv_clahe_r = []
    mapped_image_hsv_clahe_g = []
    mapped_image_hsv_clahe_b = []

    mapped_image_hsv_clahe_list = mapped_image_hsv_clahe.tolist()
    for row in mapped_image_hsv_clahe_list:
        for rgb in row:
            mapped_image_hsv_clahe_r.append(rgb[0])
            mapped_image_hsv_clahe_g.append(rgb[1])
            mapped_image_hsv_clahe_b.append(rgb[2])

    #print_mapping(image_r, mapped_image_hsv_clahe_r, method + ':R')
    #print_mapping(image_g, mapped_image_hsv_clahe_g, method + ':G')
    #print_mapping(image_b, mapped_image_hsv_clahe_b, method + ':B')
            
    return mapped_image_combined, mapped_image_rgb, mapped_image_hsv, mapped_image_hsv_clahe


def main():
    image_dict = create_image_dict()
    
    for image_name, image in image_dict.items():   
        mapped_image_combined, mapped_image_rgb, mapped_image_hsv, mapped_image_hsv_clahe = equalize_hist(image)
    
        # UNCOMMENT TO PLOT ALL IMAGES AND HISTOGRAMS
        plot_image_hist(image, image_name + ' (Original)', 'HISTOGRAM (ORIGINAL)')
        plot_image_hist(mapped_image_combined, image_name + ' (HE RGB COMBINED)', 'HISTOGRAM (RGB COMBINED)')
        plot_image_hist(mapped_image_rgb, image_name + ' (HE RGB SEPARATE)', 'HISTOGRAM (RGB SEPARATELY)')
        plot_image_hist(mapped_image_hsv, image_name + ' (HE HSV)', 'HISTOGRAM (HSV)')
        plot_image_hist(mapped_image_hsv_clahe, image_name + ' (HE CLAHE WITH HSV 4x4 GRID)', 'HISTOGRAM (CLAHE WITH HSV 4x4 GRID)')

# Run
main()    
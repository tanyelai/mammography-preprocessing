import os
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm  # Import the tqdm library

### md.ai DICOM to PNG conversion code

def convert_dicom_to_png(dicom_file: str) -> np.ndarray:

    data = pydicom.read_file(dicom_file)
    if ('WindowCenter' not in data) or\
       ('WindowWidth' not in data) or\
       ('PhotometricInterpretation' not in data) or\
       ('RescaleSlope' not in data) or\
       ('PresentationIntentType' not in data) or\
       ('RescaleIntercept' not in data):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element('PresentationIntentType').value
    if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
        print(f"{dicom_file} got processing file")
        return


    c = data.data_element('WindowCenter').value # data[0x0028, 0x1050].value
    w = data.data_element('WindowWidth').value  # data[0x0028, 0x1051].value
    if type(c)==pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    try:
        a = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get get pixel_array!')
        return

    slope = data.data_element('RescaleSlope').value
    intercept = data.data_element('RescaleIntercept').value
    a = a * slope + intercept

    try:
        pad_val = data.get('PixelPaddingValue')
        pad_limit = data.get('PixelPaddingRangeLimit', -99999)
        if pad_limit == -99999:
            mask_pad = (a==pad_val)
        else:
            if str(photometricInterpretation) == 'MONOCHROME2':
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f'{dicom_file} has no PixelPaddingValue')
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(f'{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}')
        except:
            print(f'{dicom_file} most frequent pixel value {sorted_pixels[0]}')

    # apply window
    mm = c - 0.5 - (w-1)/2
    MM = c - 0.5 + (w-1)/2
    a[a<mm] = 0
    a[a>MM] = 255
    mask = (a>=mm) & (a<=MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w-1) + 0.5) * 255

    if str( photometricInterpretation ) == 'MONOCHROME1':
        a = 255 - a

    a[mask_pad] = 0
    return a

def save_png(input_array: np.ndarray, target_file_path: Path):
    image = input_array.astype(np.uint8)
    image_pil = Image.fromarray(image)
    image_pil.save(str(target_file_path))

def process_dataset(root_directory, save_directory):
    # Ensure the saving directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    root = Path(root_directory)
    
    # Get a list of all .dcm files
    all_dicom_files = list(root.rglob('*.dcm'))
    
    # Iterate through all .dcm files in the root directory and its subdirectories with a progress bar
    for dicom_file in tqdm(all_dicom_files, desc="Processing DICOM files", unit="file"):
        # Convert the dicom file to png
        image_array = convert_dicom_to_png(str(dicom_file))
        if image_array is not None:
            png_filename = dicom_file.stem + ".png"
            save_png(image_array, Path(save_directory) / png_filename)

if __name__ == '__main__':
    # ArgumentParser to receive the root directory and saving directory
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Process DICOM images and convert to PNG.')
    parser.add_argument('-r', '--root-directory', required=True, type=str, help='Root directory of the dataset')
    parser.add_argument('-s', '--save-directory', required=True, type=str, help='Directory to save PNG files')
    
    args = parser.parse_args()
    process_dataset(args.root_directory, args.save_directory)
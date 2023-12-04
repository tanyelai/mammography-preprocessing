import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import gc
from tqdm import tqdm
from functools import partial
from itertools import islice



def adjust_bounding_boxes(bboxes, crop_xmin, crop_ymin, crop_xmax, crop_ymax, original_size, target_size=None, laterality=None):
    """
    Adjust the bounding boxes after cropping and resizing the image.
    """
    # Determine the padding added for width and height before resizing
    # This depends on whether the breast is on the left or right side of the image
    if original_size[0] > original_size[1]:  # Height is greater than width
        diff = original_size[0] - (crop_xmax - crop_xmin)
        left_pad = diff if laterality == 'R' else 0
        right_pad = diff if laterality == 'L' else 0
    else:
        left_pad = 0
        right_pad = 0

    # Calculate the new width including padding
    new_width = crop_xmax - crop_xmin + left_pad + right_pad

    # Calculate scale factors for x and y dimensions
    scale_x = target_size[0] / new_width
    scale_y = target_size[1] / (crop_ymax - crop_ymin)

    new_bboxes = []
    for bbox in bboxes:
        # Adjust the bounding box coordinates based on cropping and padding
        new_x_min = ((bbox[0] - crop_xmin) + left_pad) * scale_x
        new_y_min = (bbox[1] - crop_ymin) * scale_y
        new_x_max = ((bbox[2] - crop_xmin) + left_pad) * scale_x
        new_y_max = (bbox[3] - crop_ymin) * scale_y

        # Clamp the coordinates to the image size
        new_x_min = max(new_x_min, 0)
        new_y_min = max(new_y_min, 0)
        new_x_max = min(new_x_max, target_size[0])
        new_y_max = min(new_y_max, target_size[1])

        new_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

    return new_bboxes



def load_image(image_path, grayscale=True):
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(image_path)

def is_patch_significant(img, x, y, size, black_threshold=0.5, white_threshold=0.5):
    region = img[y:y + size, x:x + size]
    if region.size == 0:
        return False
    black_ratio = np.count_nonzero(region == 0) / region.size
    white_ratio = np.count_nonzero(region == 255) / region.size
    return black_ratio < black_threshold and white_ratio < white_threshold


def extract_largest_rectangle(img, patch_size, stride, black_threshold, laterality):
    """
    Extract the largest rectangle from the grayscale image.
    """
    h, w = img.shape
    min_x, min_y, max_x, max_y = w, h, 0, 0

    for y in range(0, h - patch_size + 1, stride):
        # For 'R' laterality, iterate from right to left
        if laterality == 'R':
            x = w - patch_size
            while x >= 0:
                if is_patch_significant(img, x, y, patch_size, black_threshold):
                    min_x, max_x = min(min_x, x), max(max_x, x + patch_size)
                    min_y, max_y = min(min_y, y), max(max_y, y + patch_size)
                x -= stride

        # For 'L' laterality, iterate from left to right
        elif laterality == 'L':
            for x in range(0, w - patch_size + 1, stride):
                if is_patch_significant(img, x, y, patch_size, black_threshold):
                    min_x, max_x = min(min_x, x), max(max_x, x + patch_size)
                    min_y, max_y = min(min_y, y), max(max_y, y + patch_size)

        # Default case, if laterality is not specified or unknown
        else:
            raise ValueError(f"Unknown laterality: {laterality}")

    largest_rectangle = img[min_y:max_y, min_x:max_x]
    return largest_rectangle, (min_x, min_y, max_x - min_x, max_y - min_y)



def save_largest_rectangle(largest_rectangle, output_path, image_id, laterality):
    rectangle_filename = f"{image_id}.png"
    rectangle_output_path = os.path.join(output_path, rectangle_filename)
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(rectangle_output_path, largest_rectangle)


def process_image(image_file, image_dir, output_dir, patch_size, stride, black_threshold, white_threshold, df_breast_annotations, target_size=None):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        img = load_image(image_path)
        image_id = image_file.split('.')[0]
        laterality = df_breast_annotations.loc[df_breast_annotations['image_id'] == image_id, 'laterality'].values[0]

        bboxes = df_breast_annotations.loc[df_breast_annotations['image_id'] == image_id, ['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        largest_rectangle, (crop_xmin, crop_ymin, crop_width, crop_height) = extract_largest_rectangle(
            img, patch_size, stride, black_threshold, laterality
        )

        original_size = (crop_height, crop_width)

        height, width = largest_rectangle.shape
        delta = abs(height - width)
        top, bottom, left, right = 0, 0, 0, 0
        
        if height > width:
            if laterality == 'L':  # More padding on the right
                left = 0
                right = delta
            else:  # More padding on the left
                left = delta
                right = 0
        else: # this case may not be even possible but still exist for outliers
            if laterality == 'L':  # More padding on the bottom
                top = 0
                bottom = delta
            else:  # More padding on the top
                top = delta
                bottom = 0

        # Apply padding
        largest_rectangle = np.pad(largest_rectangle, ((top, bottom), (left, right)), mode='constant', constant_values=0)

        # Resize if target size is specified
        if target_size:
            largest_rectangle = cv2.resize(largest_rectangle, target_size, interpolation=cv2.INTER_AREA)
            scaled_target_size = target_size
        else:
            # Keep the original size after padding
            scaled_target_size = (largest_rectangle.shape[1], largest_rectangle.shape[0])

        # Adjust the bounding boxes
        scaled_bboxes = adjust_bounding_boxes(
            bboxes, crop_xmin, crop_ymin, crop_xmin + crop_width, crop_ymin + crop_height,
            original_size, scaled_target_size, laterality
        )

        # Save the largest rectangle
        save_largest_rectangle(largest_rectangle, output_dir, image_id, laterality)

        del img, largest_rectangle
        gc.collect()
        return image_id, scaled_bboxes



def create_patch_dataset(image_dir, breast_annotations_file, output_dir, 
                         patch_size, stride, black_threshold, white_threshold, target_size=None):
    """
    Create a dataset of patches from the images in the image directory.
    """
    df_breast_annotations = pd.read_csv(breast_annotations_file)
    os.makedirs(output_dir, exist_ok=True)
    
    scaled_bboxes_dict = {}

    process_image_with_args = partial(
        process_image,
        image_dir=image_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        black_threshold=black_threshold,
        white_threshold=white_threshold,
        df_breast_annotations=df_breast_annotations,
        target_size=target_size
    )

    with ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_image_with_args, os.listdir(image_dir)), total=len(os.listdir(image_dir))):
            if result:  
                image_id, scaled_bboxes = result
                scaled_bboxes_dict[image_id] = scaled_bboxes

    # Convert the dictionary of scaled bounding boxes to a DataFrame
    scaled_bboxes_list = [
        {"image_id": image_id, "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
        for image_id, bboxes in scaled_bboxes_dict.items() for bbox in bboxes
    ]
    scaled_bboxes_df = pd.DataFrame(scaled_bboxes_list)
    scaled_bboxes_df.to_csv(f'{output_dir}/scaled_bboxes.csv', index=False)



def test(image_dir, breast_annotations_file, output_dir, patch_size, stride, black_threshold, white_threshold, num_patients=100, target_size=None):
    df_breast_annotations = pd.read_csv(breast_annotations_file)
    os.makedirs(output_dir, exist_ok=True)
    
    scaled_bboxes_dict = {}

    process_image_with_args = partial(
        process_image,
        image_dir=image_dir,
        output_dir=output_dir,
        patch_size=patch_size,
        stride=stride,
        black_threshold=black_threshold,
        white_threshold=white_threshold,
        df_breast_annotations=df_breast_annotations,
        target_size=target_size
    )

    with ThreadPoolExecutor() as executor:
        image_files = list(islice(os.listdir(image_dir), num_patients))  # Only take the first three image files
        for result in tqdm(executor.map(process_image_with_args, image_files), total=len(image_files)):
            if result:  
                image_id, scaled_bboxes = result
                scaled_bboxes_dict[image_id] = scaled_bboxes

    # Convert the dictionary of scaled bounding boxes to a DataFrame
    scaled_bboxes_list = [
        {"image_id": image_id, "xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]}
        for image_id, bboxes in scaled_bboxes_dict.items() for bbox in bboxes
    ]
    scaled_bboxes_df = pd.DataFrame(scaled_bboxes_list)
    scaled_bboxes_df.to_csv(f'{output_dir}/scaled_bboxes.csv', index=False)



# create_patch_dataset(
#     image_dir="/home/info/workspace/data/images",
#     breast_annotations_file="/home/info/workspace/annotations/finding_annotations.csv",
#     output_dir="/home/info/workspace/data/images-patchcut-1024x1024",
#     patch_size=400,
#     stride=80,
#     black_threshold=0.3,
#     white_threshold=0.5,
#     target_size=(1024, 1024) # if not specified, use the size of the largest rectangle with padding
# )



test(
    image_dir="/home/info/workspace/data/images",
    breast_annotations_file="/home/info/workspace/annotations/finding_annotations.csv",
    output_dir="/home/info/workspace/data/test",
    patch_size=400,
    stride=80,
    black_threshold=0.3,
    white_threshold=0.5,
    #target_size=(1024, 1024) # if not specified, use the size of the largest rectangle with padding
)

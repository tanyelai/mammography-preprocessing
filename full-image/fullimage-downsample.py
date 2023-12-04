import os
import random
import pandas as pd
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToPILImage


def read_annotations(csv_path, set_type):
    """
    Read the annotations CSV file and return a dictionary mapping from image filename to BI-RADS label for the specified set type.
    :param csv_path: Path to the CSV file containing annotations.
    :param set_type: Set type: 'training', 'validation' or 'test'.
    :return: Dictionary of image filename to BI-RADS label for the specified set type.
    """
    annotations = pd.read_csv(csv_path, index_col=False)
    # Filter annotations based on the 'split' column
    filtered_annotations = annotations[annotations['split'] == set_type]
    # Assuming the CSV has 'image_id' and 'BI-RADS' columns
    return dict(zip(filtered_annotations['image_id'], filtered_annotations['breast_birads']))

def downsample_and_save_data(data_dir, save_dir, set_type, annotations_path):
    data_transforms = {
        'training': transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ]),
        'validation': transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ]),
    }

    annotations = read_annotations(annotations_path, set_type)
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(data_dir) for f in filenames if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg']]
    image_labels = [annotations.get(os.path.splitext(os.path.basename(image))[0], None) for image in images]

    class_counts = {}
    class_samples = {}

    # Processing each image and its label
    for image, label in tqdm(zip(images, image_labels), desc=f"Processing {set_type} set", total=len(images)):
        if label is None:
            continue  # Skip images without a label
        if label in ["BI-RADS 2", "BI-RADS 3"]:
            continue  # Skip BI-RADS 2 and BI-RADS 3 categories
        if label in ["BI-RADS 4", "BI-RADS 5"]:
            label = "BI-RADS45"  # Merge BI-RADS 4 and BI-RADS 5

        if label not in class_counts:
            class_counts[label] = 0
            class_samples[label] = []

        class_counts[label] += 1
        class_samples[label].append(image)

    # Handle potential issue of no images for a category
    if not class_counts:
        raise ValueError("No images found for the given set type. Check your annotations and image directories.")

    # Equalize BI-RADS 1 and BI-RADS45 counts
    birads45_count = class_counts.get("BI-RADS45", 0)
    birads1_count = class_counts.get("BI-RADS 1", 0)

    # Find minimum count between BI-RADS45 and BI-RADS 1 to equalize their numbers
    min_count = min(birads45_count, birads1_count)

    # Downsample BI-RADS45 if necessary
    if birads45_count > min_count:
        class_samples["BI-RADS45"] = random.sample(class_samples["BI-RADS45"], min_count)
        class_counts["BI-RADS45"] = min_count

    # Downsample BI-RADS 1 if necessary
    if birads1_count > min_count:
        class_samples["BI-RADS 1"] = random.sample(class_samples["BI-RADS 1"], min_count)
        class_counts["BI-RADS 1"] = min_count

    save_path = os.path.join(save_dir, set_type)
    os.makedirs(save_path, exist_ok=True)

    # Save the images with tqdm progress bar for each class label
    for label, samples in class_samples.items():
        label_dir = os.path.join(save_path, label)
        os.makedirs(label_dir, exist_ok=True)

        # Wrap the samples iterable with tqdm for progress reporting
        for sample in tqdm(samples, desc=f"Saving images for {label}", total=len(samples)):
            image = Image.open(sample)
            image_transformed = data_transforms[set_type](image)
            # Convert tensor back to PIL Image to save it as an image file
            image_to_save = ToPILImage()(image_transformed)
            image_path = os.path.join(label_dir, os.path.basename(sample))
            image_to_save.save(image_path)  # Now we're saving a PIL Image, not a Tensor



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="downsample and save data.")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Directory containing the data.")
    parser.add_argument("-s", "--save_dir", type=str, required=True, help="Directory to save the downsampled data.")
    parser.add_argument("-t", "--set_type", type=str, required=True, help="Set type: 'training', 'validation' or 'test'.")
    parser.add_argument("-a", "--annotations_path", type=str, required=True, help="Path to the CSV file with annotations.")
    args = parser.parse_args()

    downsample_and_save_data(args.data_dir, args.save_dir, args.set_type, args.annotations_path)
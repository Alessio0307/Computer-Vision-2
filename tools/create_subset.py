import os
import json
import random
import zipfile
from shutil import copy2, rmtree
from tqdm import tqdm

def reduce_coco_subset(zip_path, output_dir, subset_fraction=0.1):
    extract_dir = os.path.join(output_dir, 'coco_subset_temp')
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    annotations_dir = os.path.join(extract_dir, 'annotations')
    train_images_dir = os.path.join(extract_dir, 'train2017')
    val_images_dir = os.path.join(extract_dir, 'val2017')

    output_annotations_dir = os.path.join(output_dir, 'coco_subset', 'annotations')
    output_train_images_dir = os.path.join(output_dir, 'coco_subset','images', 'subset_train2017')
    output_val_images_dir = os.path.join(output_dir, 'coco_subset', 'images', 'subset_val2017')

    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_val_images_dir, exist_ok=True)

    with open(os.path.join(annotations_dir, 'person_keypoints_train2017.json'), 'r') as f:
        train_annotations = json.load(f)
    with open(os.path.join(annotations_dir, 'person_keypoints_val2017.json'), 'r') as f:
        val_annotations = json.load(f)

    train_images = train_annotations['images']
    val_images = val_annotations['images']
    
    train_subset_size = int(len(train_images) * subset_fraction)
    val_subset_size = int(len(val_images) * subset_fraction)
    
    train_subset = random.sample(train_images, train_subset_size)
    val_subset = random.sample(val_images, val_subset_size)

    train_image_ids = {image['id'] for image in train_subset}
    val_image_ids = {image['id'] for image in val_subset}

    train_annotations_subset = {
        'info': train_annotations['info'],
        'licenses': train_annotations['licenses'],
        'images': train_subset,
        'annotations': [anno for anno in train_annotations['annotations'] if anno['image_id'] in train_image_ids],
        'categories': train_annotations['categories']
    }
    val_annotations_subset = {
        'info': val_annotations['info'],
        'licenses': val_annotations['licenses'],
        'images': val_subset,
        'annotations': [anno for anno in val_annotations['annotations'] if anno['image_id'] in val_image_ids],
        'categories': val_annotations['categories']
    }

    with open(os.path.join(output_annotations_dir, 'person_keypoints_subset_train2017.json'), 'w') as f:
        json.dump(train_annotations_subset, f, indent=4)
    with open(os.path.join(output_annotations_dir, 'person_keypoints_subset_val2017.json'), 'w') as f:
        json.dump(val_annotations_subset, f, indent=4)

    print("Copying training images...")
    for image in tqdm(train_subset, desc="Training Images"):
        src = os.path.join(train_images_dir, image['file_name'])
        dst = os.path.join(output_train_images_dir, image['file_name'])
        if os.path.exists(src):
            copy2(src, dst)
        else:
            print(f"File not found: {src}")

    print("Copying validation images...")
    for image in tqdm(val_subset, desc="Validation Images"):
        src = os.path.join(val_images_dir, image['file_name'])
        dst = os.path.join(output_val_images_dir, image['file_name'])
        if os.path.exists(src):
            copy2(src, dst)
        else:
            print(f"File not found: {src}")

    rmtree(extract_dir)

    filter_annotations(output_train_images_dir, os.path.join(output_annotations_dir, 'person_keypoints_subset_train2017.json'))
    filter_annotations(output_val_images_dir, os.path.join(output_annotations_dir, 'person_keypoints_subset_val2017.json'))

    print("Subset created in:", output_dir)

def get_image_ids(image_dir):
    image_ids = {}
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            image_id = int(os.path.splitext(image_name)[0])
            image_ids[image_id] = image_name
    return image_ids

def filter_annotations(image_dir, annotation_file):
    image_ids = get_image_ids(image_dir)
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    new_annotations = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": [],
        "annotations": [],
        "categories": coco_data.get("categories", [])
    }

    image_id_map = {}
    for image_info in coco_data["images"]:
        if image_info["id"] in image_ids:
            new_annotations["images"].append(image_info)
            image_id_map[image_info["id"]] = image_info["id"]
    
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] in image_id_map:
            new_annotations["annotations"].append(annotation)
    
    with open(annotation_file, 'w') as f:
        json.dump(new_annotations, f, indent=4)


# Define paths
zip_path = '/mnt/d/Download/coco2017.zip'
output_dir = '/mnt/d/Download/'

# Create subset
reduce_coco_subset(zip_path, output_dir, subset_fraction=0.1)

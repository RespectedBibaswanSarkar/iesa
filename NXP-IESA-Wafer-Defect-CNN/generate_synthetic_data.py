import cv2
import numpy as np
import os
import random
import argparse
import shutil
import zipfile
from tqdm import tqdm

# Configuration
CLASSES = ["Bridge", "CMP", "Cracks", "Opens", "LER", "Vias"]
IMG_SIZE = 256
NUM_IMAGES_PER_CLASS = 100 

def add_sem_noise(image):
    row, col = image.shape
    mean = 0
    var = 50
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    noisy = cv2.GaussianBlur(noisy, (3, 3), 0.5)
    return noisy

def generate_background():
    bg = np.full((IMG_SIZE, IMG_SIZE), 120, dtype=np.uint8)
    noise = np.random.randint(-10, 10, (IMG_SIZE, IMG_SIZE))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return bg

def draw_lines(img, num_lines=3, thickness=20):
    h, w = img.shape
    step = w // (num_lines + 1)
    centers = []
    for i in range(1, num_lines + 1):
        x = i * step
        cv2.rectangle(img, (x - thickness//2, 0), (x + thickness//2, h), 200, -1)
        centers.append(x)
    return img, centers

def generate_bridge(img, line_centers):
    if len(line_centers) < 2: return img
    idx = random.randint(0, len(line_centers) - 2)
    x1 = line_centers[idx]
    x2 = line_centers[idx+1]
    y = random.randint(50, IMG_SIZE - 50)
    cv2.line(img, (x1, y), (x2, y), 180, random.randint(5, 15))
    return img

def generate_cmp(img):
    for _ in range(random.randint(3, 8)):
        x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        cv2.line(img, (x1, y1), (x2, y2), 80, 1) 
    for _ in range(2):
        center = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        radius = random.randint(20, 50)
        mask = np.zeros_like(img)
        cv2.circle(mask, center, radius, 1, -1)
        img = np.where(mask==1, cv2.add(img, -20), img)
    return img

def generate_cracks(img):
    x, y = random.randint(50, 200), random.randint(50, 200)
    pts = [(x, y)]
    for _ in range(10):
        x += random.randint(-10, 10)
        y += random.randint(-10, 10)
        pts.append((x, y))
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], 40, 2)
    return img

def generate_opens(img, line_centers):
    if not line_centers: return img
    x = random.choice(line_centers)
    y = random.randint(50, IMG_SIZE - 50)
    h = random.randint(10, 30)
    cv2.rectangle(img, (x - 15, y), (x + 15, y + h), 120, -1)
    return img

def generate_ler(img, line_centers):
    img = generate_background()
    h, w = img.shape
    thickness = 20
    for x_center in line_centers:
        pts_left = []
        pts_right = []
        for y in range(0, h, 5):
            offset = random.randint(-3, 3)
            pts_left.append((x_center - thickness//2 + offset, y))
            pts_right.append((x_center + thickness//2 + offset, y))
        poly = np.array(pts_left + pts_right[::-1], np.int32)
        cv2.fillPoly(img, [poly], 200)
    return img

def generate_vias(img):
    img = generate_background() 
    for _ in range(random.randint(5, 10)):
        center = (random.randint(20, IMG_SIZE-20), random.randint(20, IMG_SIZE-20))
        cv2.circle(img, center, 15, 40, -1) 
        cv2.circle(img, center, 25, 180, 2) 
    return img

def zip_dataset(source_dir, zip_name):
    print(f"Zipping {source_dir} to {zip_name}...")
    shutil.make_archive(zip_name.replace('.zip', ''), 'zip', source_dir)
    print("Zip created.")

def main(args):
    print(f"Generating dataset in {args.output_dir}...")
    
    for class_name in CLASSES:
        output_path = os.path.join(args.output_dir, class_name)
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Generating {class_name}...")
        for i in tqdm(range(args.num_images)):
            img = generate_background()
            
            if class_name in ["Bridge", "Opens", "LER"]:
                img, centers = draw_lines(img)
                if class_name == "Bridge":
                    img = generate_bridge(img, centers)
                elif class_name == "Opens":
                    img = generate_opens(img, centers)
                elif class_name == "LER":
                    img = generate_ler(img, centers)
            elif class_name == "CMP":
                 img, _ = draw_lines(img)
                 img = generate_cmp(img)
            elif class_name == "Cracks":
                img = generate_cracks(img)
            elif class_name == "Vias":
                img = generate_vias(img)

            img = add_sem_noise(img)
            filename = f"{class_name.lower()}_{i:04d}.png"
            cv2.imwrite(os.path.join(output_path, filename), img)
            
    if args.zip_name:
        zip_dataset(args.output_dir, args.zip_name)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="dataset", help="Dataset directory")
    parser.add_argument("--num_images", type=int, default=100, help="Images per class")
    parser.add_argument("--zip_name", help="Name of zip file to create (e.g. dataset_v2.zip)")
    
    args = parser.parse_args()
    main(args)

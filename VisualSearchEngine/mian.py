import csv
import urllib
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import faiss
import validators

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

def initialize_faiss_index(dimension):
  index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
  return index

# Load YOLOX model
def load_model():
  exp = get_exp('/Users/mpichikala/personal/YOLOX/exps/default/yolox_s.py', None)  # YOLOX-s model
  model = exp.get_model()
  model.eval()
  ckpt_file = "/Users/mpichikala/personal/AIProject/VisualSearchEngine/yolox_s.pth"  # Pretrained weights file
  ckpt = torch.load(ckpt_file, map_location="mps")
  model.load_state_dict(ckpt["model"])
  model = fuse_model(model)  # Fuse layers for inference speed
  return model, exp

# Read csv file and print its content
def read_image_urls_from_csv(file_path):
  csv_file = open(file_path, 'r')
  csv_reader = csv.reader(csv_file)
  urls = []
  for row in csv_reader:
    if validators.url(row[1]):
      urls.append(row[1])
  return urls

def load_image_from_url(url):
  try:
    response = urllib.request.urlopen(url)
    img = Image.open(BytesIO(response.read())).convert('RGB')
    img = np.array(img)
    return img
  except Exception as e:
    print(f"Failed to load image from {url} because of {e}")
    return None
  
def preprocess_img(img):
  img, ratio = preproc(img, 0)
  img = torch.from_numpy(img).unsqueeze(0).float()  # Add batch dimension
  return img, ratio

# Preprocess image for YOLOX
def preprocess_img(img, exp):
  img, ratio = preproc(img, exp.test_size)
  img = torch.from_numpy(img).unsqueeze(0).float()  # Add batch dimension
  return img, ratio

# Detect objects in an image
def detect_objects(model, img, exp):
  with torch.no_grad():
    outputs = model(img)
  # Ensure you're unpacking properly
  if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
    # Unpack boxes and scores if YOLOX returns multiple values
    boxes, scores = outputs[0], outputs[1]
    return boxes, scores
  else:
    return outputs  # Return as is if it's a single value

def extract_features_from_boxes(outputs):
  if outputs is not None and len(outputs) > 0:
    boxes = outputs[:, :4].cpu().numpy()  # x1, y1, x2, y2 for bounding boxes
    centroids = np.mean(boxes, axis=1)  # Calculate centroids (mean of x1, y1, x2, y2)
    return centroids
  else:
    return np.array([])  # Return an empty array if no objects are detected

# Add embeddings to FAISS index
def add_embeddings_to_faiss(index, embeddings):
  index.add(embeddings)

def process_and_index_images(image_urls, model, exp, index):
  for url in image_urls:
    image = load_image_from_url(url)
    if image is not None:
      image, _ = preprocess_img(image, exp)
      outputs = detect_objects(model, image, exp)

      # Extract embeddings (use bounding box centroids)
      embeddings = extract_features_from_boxes(outputs)
      # Add embeddings to FAISS index
      add_embeddings_to_faiss(index, embeddings)
    else:
      print(f"Skipping image from URL: {url}")

# Search for nearest embeddings in FAISS
def search_faiss(index, query_embedding, top_k=5):
  distances, indices = index.search(query_embedding, top_k)
  return distances, indices

# Query an image URL and search for similar objects
def query_image(img_url, model, exp, index):
  img = load_image_from_url(img_url)  # Load image from URL
  if img is not None:
    img, _ = preprocess_img(img, exp)
    outputs = detect_objects(model, img, exp)
    
    query_embedding = extract_features_from_boxes(outputs)
    
    # Search the FAISS index
    distances, indices = search_faiss(index, query_embedding)
    
    print(f"Found {len(indices)} similar objects")
    return distances, indices
  else:
    print(f"Failed to process query image from URL: {img_url}")
    return None, None

def main():
  CSV_PATH = '/Users/mpichikala/personal/AIProject/VisualSearchEngine/train.csv'
  
  # Load YOLOX model and initialize FAISS index
  model, exp = load_model()
  
  index = initialize_faiss_index(dimension=4)  # Centroid embeddings have dimension 4

  image_urls = read_image_urls_from_csv(CSV_PATH)
  process_and_index_images(image_urls, model, exp, index)
  
  # Query for similar objects using a new image URL
  query_url = "https://upload.wikimedia.org/wikipedia/commons/4/45/Livadiiskyi_Park%2C.JPG"
  distances, indices = query_image(query_url, model, exp, index)
  
  if distances is not None:
    print(f"Search Results: {indices}")

if __name__ == "__main__":
  main()

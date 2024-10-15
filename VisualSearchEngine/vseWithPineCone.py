import csv
import urllib
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import pinecone
import validators
from pinecone import Pinecone, ServerlessSpec

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# Initialize Pinecone
def initialize_pinecone_index(index_name, dimension):
  pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")
  # pc.create_index(name="image-search-index", dimension=85, 
  #   spec=ServerlessSpec(cloud='aws', region='us-east-1')
  # )
  index = pc.Index("image-search-index")
  return index

# pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")
# index = pc.Index("image-search-index")

# Load YOLOX model
def load_model(device):
  exp = get_exp('/Users/mpichikala/personal/YOLOX/exps/default/yolox_s.py', None)  # YOLOX-s model
  model = exp.get_model()
  model.eval()
  ckpt_file = "/Users/mpichikala/personal/AIProject/VisualSearchEngine/yolox_s.pth"  # Pretrained weights file
  ckpt = torch.load(ckpt_file, map_location="mps")
  model.load_state_dict(ckpt["model"])
  model = fuse_model(model)  # Fuse layers for inference speed
  model = model  # Move the model to MPS device
  return model, exp

# Read csv file and extract valid image URLs
def read_image_urls_from_csv(file_path):
  with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    urls = [row[1] for row in csv_reader if validators.url(row[1])]
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

# Preprocess image for YOLOX
def preprocess_img(img, exp, device):
  img, ratio = preproc(img, exp.test_size)
  img = torch.from_numpy(img).unsqueeze(0).float()  # Add batch dimension
  img = img.to(device)  # Move image to MPS device
  img = img.to('cpu')  # Convert back to CPU tensor if necessary for further processing
  return img, ratio

# Detect objects in an image
# Detect objects in an image
def detect_objects(model, img, exp, device):
  # img = img.to(device)  # Ensure the image is on the correct device (MPS or CPU)

  with torch.no_grad():
    outputs = model(img)

  # Move outputs to CPU to avoid issues with MPS tensors
  if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
    boxes, scores = outputs[0], outputs[1]  # Move tensors back to CPU
    return boxes, scores
  else:
    return outputs  # Return as is if it's a single value, and move to CPU

# Extract features (centroids of bounding boxes)
def extract_features_from_boxes(outputs):
  if outputs is not None and len(outputs) > 0:
    boxes = outputs[:, :4].cpu().numpy()  # x1, y1, x2, y2 for bounding boxes
    centroids = np.mean(boxes, axis=1)  # Calculate centroids
    return centroids
  else:
    return np.array([])  # Return an empty array if no objects are detected

# Add embeddings to Pinecone index
def add_embeddings_to_pinecone(index, embeddings, ids):
  # embeddings should be in shape (n_samples, embedding_dimension)
  # ids should be a list of unique identifiers for each embedding
  vectors = [(str(id), embedding) for id, embedding in zip(ids, embeddings)]
  index.upsert(vectors)

def process_and_index_images(image_urls, model, exp, index, device):
  for i, url in enumerate(image_urls):
    if i >= 1522 and i < 1540:
      print(f"processing url {url}")
      image = load_image_from_url(url)
      if image is not None:
        image, _ = preprocess_img(image, exp, device)
        outputs = detect_objects(model, image, exp, device)
        embeddings = extract_features_from_boxes(outputs)
        if embeddings.size > 0:
          # Add embeddings to Pinecone with unique IDs
          embeddings = torch.tensor(embeddings).to("cpu").numpy()
          print(f"inserting to pine at index {i} of {url}")
          add_embeddings_to_pinecone(index, embeddings, [i])
      else:
        print(f"Skipping image from URL: {url}")

# Search for nearest embeddings in Pinecone
def search_pinecone(index, query_embedding, top_k=5):
  result = index.query(vector=query_embedding.tolist(), top_k=top_k)
  return result

# Query an image URL and search for similar objects
def query_image(img_url, model, exp, index, device, top_k=5):
  img = load_image_from_url(img_url)
  if img is not None:
      img, _ = preprocess_img(img, exp, device)
      outputs = detect_objects(model, img, exp, device)
      query_embedding = extract_features_from_boxes(outputs)
      if query_embedding.size > 0:
        query_embedding = torch.tensor(query_embedding).to("cpu").numpy()
        result = search_pinecone(index, query_embedding, top_k=top_k)
        print(f"Found {len(result['matches'])} similar objects")
        return result
  else:
    print(f"Failed to process query image from URL: {img_url}")
    return None

def main():
  CSV_PATH = '/Users/mpichikala/personal/AIProject/VisualSearchEngine/train.csv'
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  # Load YOLOX model and initialize Pinecone index
  model, exp = load_model(device)
  index_name = "image-search-index"
  index = initialize_pinecone_index(index_name=index_name, dimension=4)  # Embedding dimension is 4 (centroid)
  
  image_urls = read_image_urls_from_csv(CSV_PATH)
  process_and_index_images(image_urls, model, exp, index, device)
  
  # Query for similar objects using a new image URL
  query_url = "https://upload.wikimedia.org/wikipedia/commons/4/45/Livadiiskyi_Park%2C.JPG"
  result = query_image(query_url, model, exp, index, device, top_k=5)
  
  if result:
    for match in result['matches']:
      print(f"ID: {match['id']}, Score: {match['score']}")

if __name__ == "__main__":
  main()

import os
import shutil
from pathlib import Path

file_extensions = set()
HOME = os.environ['HOME']
DOWNLOADS_PATH = Path(HOME + '/Downloads')

def main():
  for filename in os.listdir(DOWNLOADS_PATH):
    if os.path.isfile(os.path.join(DOWNLOADS_PATH, filename)):
      _, extension = os.path.splitext(filename)
      file_extensions.add(extension)

  for extension in file_extensions:
    # create a folder for each file extension
    if extension == '':
      folder_path = os.path.join(DOWNLOADS_PATH, 'other-files')
    else:
      folder_path = os.path.join(DOWNLOADS_PATH, extension.split('.')[1] + '-files')
    os.makedirs(folder_path, exist_ok = True)

  # move files to their respective folders
  for filename in os.listdir(DOWNLOADS_PATH):
    if os.path.isfile(os.path.join(DOWNLOADS_PATH, filename)):
      _, extension = os.path.splitext(filename)
      print(filename)
      source_path = os.path.join(DOWNLOADS_PATH, filename)
      if extension == '':
        destination_path = os.path.join(DOWNLOADS_PATH, 'other-files')
      else:
        destination_path = os.path.join(DOWNLOADS_PATH, extension.split('.')[1] + '-files')

      if os.path.exists(os.path.join(destination_path, filename)):
        print("File already exists in destination folder.")
        continue
      shutil.move(source_path, destination_path)

  print("File Moving done successfully!")
  
def get_latest_file_timestamp(subdir_path):
  """Get the latest modified timestamp of all files in the subdir."""
  file_paths = []
  for root, dirs, files in os.walk(subdir_path):
    for file in files:
      file_paths.append(os.path.join(root, file))

  if not file_paths:
    return None  # Return None if subdir has no files

  latest_time = max(os.path.getmtime(file) for file in file_paths)
  return latest_time

def sort_subdirs_by_latest_file(directory):
  # Get all subdirectories inside the given directory
  subdirs = [os.path.join(directory, subdir) for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
  
  # Create a list of tuples (subdir, latest_timestamp)
  subdirs_with_latest_time = []
  for subdir in subdirs:
      latest_timestamp = get_latest_file_timestamp(subdir)
      if latest_timestamp is not None:
          subdirs_with_latest_time.append((subdir, latest_timestamp))
  
  # Sort subdirectories by timestamp (latest first)
  sorted_subdirs = sorted(subdirs_with_latest_time, key=lambda x: x[1], reverse=True)
  
  return [subdir for subdir, _ in sorted_subdirs]

if __name__ == "__main__":
  main()
  # Example usage
  directory = DOWNLOADS_PATH  # Replace with your directory path
  sorted_subdirs = sort_subdirs_by_latest_file(directory)

  # Show the latest subdirectory on top
  if sorted_subdirs:
    print(f"The latest subdirectory is: {sorted_subdirs[0]}")
  else:
    print("No subdirectories with files found.")

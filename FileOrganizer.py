import os
from pathlib import Path
import shutil

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

if __name__ == "__main__":
  main()

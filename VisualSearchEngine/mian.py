import csv
from yolox.data.data_augment import preproc

CSV_PATH = 'train.csv'

# Read csv file and print its content
def csv_reader():
  csv_file = open(CSV_PATH, 'r')
  csv_reader = csv.reader(csv_file)
  for row in csv_reader:
    print(row[1])

csv_reader()

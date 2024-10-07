from transformers import pipeline
from PIL import Image

pipe = pipeline("document-question-answering", model="naver-clova-ix/donut-base-finetuned-docvqa", device='mps')

# single question at a item
question = "What is the quantity of Item 3?"
image = Image.open("data/img.png")

print(pipe(image=image, question=question))

## [{'answer': '20,000$'}]

from transformers import pipeline, AutoTokenizer
import torch
from datasets import load_dataset, Audio

# classifier = pipeline('sentiment-analysis', device='mps')
# print(classifier("We are very happy to show you the 🤗 Transformers library."))
# results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
#
# speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device='mps')
# dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
# dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
# result = speech_recognizer(dataset[:4]["audio"])
# print([d["text"] for d in result])

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
encoded_data = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_data)

decoded_data = tokenizer.decode(encoded_data["input_ids"])
print(decoded_data)

batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]

encoded_data = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True)
print(encoded_data)

decoded_data = tokenizer.batch_decode(encoded_data["input_ids"])
print(decoded_data)

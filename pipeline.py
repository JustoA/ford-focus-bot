import pytesseract
from transformers import pipeline
import torch
from datasets import load_dataset, Image
from tqdm.auto import tqdm
from transformers.pipelines.base import KeyDataset

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
dataset = load_dataset("imagefolder", data_dir=r'image_base')

extractive_pipe = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa", device='cuda'
)
answers = []
print(dataset.values())
for img in tqdm(dataset['train']['image']):
    result = extractive_pipe(img.filename, "What is a break lamp bulb fault?")
    print(result)
    answers.append(result[0])
answers = sorted(answers, key=lambda s: s['score'], reverse=True)
print(answers)

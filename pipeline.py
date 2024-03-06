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
for img in tqdm(dataset['train']['image'][110:120]): #Answer is on page 114
    result = extractive_pipe(img, "What is the action / description next to the lighting message \"break lamp bulb fault?\"", max_answer_len=50)
    answers = answers + result if result[0]['answer'] is not None else answers
answers = sorted(answers, key=lambda s: s['score'], reverse=True)
print(answers)

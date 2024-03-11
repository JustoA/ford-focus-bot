from pprint import pprint

import pytesseract
from transformers import pipeline
import torch
from datasets import load_dataset, Image
from tqdm.auto import tqdm
from transformers.pipelines.base import KeyDataset

def query(imageset, prompt, pipe):
    guesses = []
    for img in tqdm(imageset):  # Answer is on page 114
        result = pipe(img,
                      prompt,
                      max_answer_len=25)
        guesses = guesses + result
    for i in range(len(guesses)):
        guesses[i]['prompt'] = prompt
    return guesses


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
dataset = load_dataset("imagefolder", data_dir=r'image_base')

extractive_pipe = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa", device='cuda'
)

answers = []
ds = dataset['train']['image'][110:120]

answers += query(ds, "What causes break lamp bulb fault?", extractive_pipe)
answers += query(ds, "Why does my screen say break lamp bulb fault?", extractive_pipe)
answers += query(ds, "What is break lamp bulb fault?", extractive_pipe)

answers = sorted(answers, key=lambda s: s['score'], reverse=True)

pprint(answers, sort_dicts=False)

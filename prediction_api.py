from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import re
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import textstat

app = FastAPI()

# Определение моделей входных и выходных данных с использованием Pydantic
class InputItem(BaseModel):
    ID: str
    Date: str
    Price: str
    Content: str

class OutputItem(BaseModel):
    ID: str
    Category: str

# Загрузка моделей и токенизатора при запуске приложения
@app.on_event("startup")
def load_models():
    global tokenizer
    global model
    global loaded_automl

    # Инициализация токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained("sergeyzh/rubert-tiny-turbo")
    model = AutoModel.from_pretrained("sergeyzh/rubert-tiny-turbo")

    # Загрузка предобученной модели AutoML
    with open('automl_model.pkl', 'rb') as f:
        loaded_automl = pickle.load(f)

# Функция для получения эмбеддингов из текста
def get_embeddings(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Функция для генерации текстовых характеристик с использованием textstat
def get_textstat_features(text):
    features = {}
    if not isinstance(text, str):
        return features
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['smog_index'] = textstat.smog_index(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    features['coleman_liau_index'] = textstat.coleman_liau_index(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)
    features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    features['difficult_words'] = textstat.difficult_words(text)
    features['linsear_write_formula'] = textstat.linsear_write_formula(text)
    features['gunning_fog'] = textstat.gunning_fog(text)
    features['text_standard'] = textstat.text_standard(text, float_output=True)
    features['syllable_count'] = textstat.syllable_count(text)
    features['lexicon_count'] = textstat.lexicon_count(text)
    features['sentence_count'] = textstat.sentence_count(text)
    return features

# Функция для извлечения признаков из DataFrame
def extract_features(df):
    # Предобработка текста: приведение к нижнему регистру
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        return text.lower()
    df['Content'] = df['Content'].apply(preprocess_text)

    # Замена номеров договоров на [CONTRACT_NUMBER]
    def replace_contract_numbers(text):
        if not isinstance(text, str):
            return text
        contract_pattern = r'\b\d{2,}-\d{4,}\b'
        text = re.sub(contract_pattern, '[CONTRACT_NUMBER]', text)
        return text
    df['Content'] = df['Content'].apply(replace_contract_numbers)

    # Замена дат на [DATE]
    def replace_dates(text):
        if not isinstance(text, str):
            return text
        date_pattern = r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'
        text = re.sub(date_pattern, '[DATE]', text)
        return text
    df['Content'] = df['Content'].apply(replace_dates)

    # Получение эмбеддингов для текста
    df['Embeddings'] = df['Content'].apply(get_embeddings)
    embeddings = df['Embeddings'].tolist()
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
    df = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
    df = df.drop(columns=['Embeddings'])

    # Генерация характеристик текста с использованием textstat
    textstat_features = df['Content'].apply(get_textstat_features)
    textstat_df = pd.DataFrame(textstat_features.tolist())
    df = pd.concat([df.reset_index(drop=True), textstat_df.reset_index(drop=True)], axis=1)

    # Определение наличия номера договора и количества слов
    def has_contract_number(text):
        if not isinstance(text, str):
            return False
        contract_pattern = r'\b\d{2,}-\d{4,}\b'
        return bool(re.search(contract_pattern, text))

    df['has_contract_number'] = df['Content'].apply(has_contract_number)
    df['word_count'] = df['Content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    # Удаление столбца 'Content', если он больше не нужен
    df = df.drop(columns=['Content'])

    return df

# Определение API-эндпоинта для предсказаний
@app.post("/predict", response_model=List[OutputItem])
def predict(inputs: List[InputItem]):
    # Преобразование входных данных в DataFrame
    data = pd.DataFrame([input.dict() for input in inputs])

    # Преобразование столбца 'Date' в формат datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y', errors='coerce')

    # Обработка столбца 'Price'
    data['Price'] = data['Price'].replace({',': '.', '-': '.'}, regex=True)
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    # Извлечение признаков
    data = extract_features(data)

    # Выполнение предсказаний с использованием загруженной модели
    predictions = loaded_automl.predict(data)

    # Получение меток классов из модели
    class_labels = loaded_automl.outer_pipes[0].ml_algos[0].models[0][0].reader.class_mapping
    label_class = {ind: class_ for class_, ind in class_labels.items()}

    # Получение предсказанных классов
    predicted_indices = np.argmax(predictions.data, axis=1)
    predicted_classes = [label_class[idx] for idx in predicted_indices]

    # Подготовка выходных данных
    output = [{'ID': id_, 'Category': category} for id_, category in zip(data['ID'], predicted_classes)]

    return output

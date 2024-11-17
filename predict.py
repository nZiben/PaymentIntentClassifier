import pandas as pd
import numpy as np
import re
from datetime import datetime
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import textstat
from tqdm.auto import tqdm
from pathlib import Path
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
tqdm.pandas()

# Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained("sergeyzh/rubert-tiny-turbo")
model = AutoModel.from_pretrained("sergeyzh/rubert-tiny-turbo")

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

def preprocess_text(text):
    # Example preprocessing function (to be replaced with your implementation)
    return text.lower()


def extract_first_date(text):
    """
    Extracts the first date found in a string and returns it as a datetime object.
    Supported formats: DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, and two-digit years.

    Args:
        text (str): The input string containing a date.

    Returns:
        datetime or None: The extracted date as a datetime object, or None if no date is found.
    """
    if not isinstance(text, str):
        return None

    # Regular expression for dates
    date_pattern = r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'
    match = re.search(date_pattern, text)

    if match:
        date_str = match.group()
        for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

    return None

    # Calculate days difference between extracted_date and the date column
def calculate_days_difference(extracted_date, reference_date):
    if pd.isna(extracted_date) or pd.isna(reference_date):
        return None
    return (reference_date - extracted_date).days


def replace_contract_numbers(text):
    if not isinstance(text, str):
        return text, False
    contract_pattern = r'(\b\d{2,}-\d{4,}\b)'
    match = re.search(contract_pattern, text)
    if match:
        return re.sub(contract_pattern, '[CONTRACT_NUMBER]', text), True
    return text, False

# Extract dates and replace them in the text
def replace_dates_and_extract(text):
    if not isinstance(text, str):
        return text, None
    date_pattern = r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'
    match = re.search(date_pattern, text)
    if match:
        extracted_date = extract_first_date(match.group())
        text = re.sub(date_pattern, '[DATE]', text)
        return text, extracted_date
    return text, None


# Функция для извлечения признаков из DataFrame
def extract_features(df):
    # Предобработка текста
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
    df["Date"], extracted_dates = zip(
        *df["Date"].apply(replace_dates_and_extract))

    df['Date'] = pd.to_datetime(df['Date'])
    df['date_difference'] = [
        calculate_days_difference(extracted_date, reference_date)
        for extracted_date, reference_date in zip(extracted_dates, df['Date'])
    ]

    # Получение эмбеддингов
    df['Embeddings'] = df['Content'].apply(get_embeddings)
    embeddings = df['Embeddings'].tolist()
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
    df = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)
    df = df.drop(columns=['Embeddings'])

    # Получение текстовых характеристик
    textstat_features = df['Content'].apply(get_textstat_features)
    textstat_df = pd.DataFrame(textstat_features.tolist())
    df = pd.concat([df.reset_index(drop=True), textstat_df.reset_index(drop=True)], axis=1)

    # Вычисление наличия номера договора и количества слов
    def has_contract_number(text):
        if not isinstance(text, str):
            return False
        contract_pattern = r'\b\d{2,}-\d{4,}\b'
        return bool(re.search(contract_pattern, text))

    df['has_contract_number'] = df['Content'].apply(has_contract_number)
    df['word_count'] = df['Content'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    # Удаление столбца 'Content' если не нужен
    df = df.drop(columns=['Content'])

    return df

# Загрузка данных из входного файла TSV без заголовков
data = pd.read_csv('input_file.tsv', sep='\t', header=None, names=['ID', 'Date', 'Price', 'Content'])

# Преобразование столбца Date в datetime
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y', errors='coerce')

# Преобразование столбца Price в числовой формат
data['Price'] = data['Price'].replace({',': '.', '-': '.'}, regex=True)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

# Извлечение признаков
print("--> Генерация признаков")
data = extract_features(data)

# Загрузка предобученной модели
print("--> Загрузка модели")
with open('models/automl_model.pkl', 'rb') as f:
    loaded_automl = pickle.load(f)

# Предсказание
print("--> Предсказание")
predictions = loaded_automl.predict(data)

# Получение меток классов
class_labels = loaded_automl.outer_pipes[0].ml_algos[0].models[0][0].reader.class_mapping
label_class = {ind: class_ for class_, ind in class_labels.items()}

# Получение предсказанных классов
predicted_indices = np.argmax(predictions.data, axis=1)
predicted_classes = [label_class[idx] for idx in predicted_indices]

# Подготовка выходного файла
output_df = pd.DataFrame({'ID': data['ID'], 'Category': predicted_classes})

# Сохранение результатов в файл TSV без заголовков
output_df.to_csv('output_file.tsv', sep='\t', index=False, header=False)
print(f"--> Файл сохранен: {Path.cwd()}")

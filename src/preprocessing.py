import re
from typing import List
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sergeyzh/rubert-tiny-turbo")
model = AutoModel.from_pretrained("sergeyzh/rubert-tiny-turbo")


def read_data(path: str, names: List[str] = None):
    data = pd.read_csv(
        path, names=names, sep='\t')
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    data['price'] = data['price'].replace({',': '.', '-': '.'}, regex=True)
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data['price'] = data['price'].apply(lambda x: float(x))

    return data


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


def extract_features(df, text_column):
    # Preprocess text
    df[text_column] = df[text_column].apply(preprocess_text)
    # Extract contract numbers and replace them in the text

    def replace_contract_numbers(text):
        if not isinstance(text, str):
            return text, False
        contract_pattern = r'(\b\d{2,}-\d{4,}\b)'
        match = re.search(contract_pattern, text)
        if match:
            return re.sub(contract_pattern, '[CONTRACT_NUMBER]', text), True
        return text, False

    df[text_column], has_contract_number = zip(
        *df[text_column].apply(replace_contract_numbers))

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

    df[text_column], extracted_dates = zip(
        *df[text_column].apply(replace_dates_and_extract))

    def get_embeddings(text: str):
        inputs = tokenizer(text, return_tensors='pt',
                           truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Calculate days difference between extracted_date and the date column
    def calculate_days_difference(extracted_date, reference_date):
        if pd.isna(extracted_date) or pd.isna(reference_date):
            return None
        return (reference_date - extracted_date).days

    df['date'] = pd.to_datetime(df['date'])
    df['date_difference'] = [
        calculate_days_difference(extracted_date, reference_date)
        for extracted_date, reference_date in zip(extracted_dates, df['date'])
    ]
    embeddings = df[text_column].apply(get_embeddings)

    # Expand embeddings into individual columns
    embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index)
    embeddings_df.columns = [f'embedding_{
        i}' for i in range(embeddings_df.shape[1])]

    # Count words in text
    word_count = df[text_column].apply(lambda x: len(x.split()))

    # Concatenate features back to the original DataFrame
    df = pd.concat([df, pd.DataFrame({
        'has_contract_number': has_contract_number,
        'word_count': word_count
    }), embeddings_df], axis=1)
    df = df.drop(columns=[text_column])
    return df


def get_dataset(path: str, cols: List[str] = None):
    data = read_data(path, cols)
    data = extract_features(data, 'content')
    return data

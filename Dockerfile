# Указываем базовый образ с Python 3.9
FROM python:3.11-slim-buster

# Устанавливаем необходимые системные пакеты
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        wget \
        ca-certificates \
        git \
        curl \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libbz2-dev \
        liblzma-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libreadline-dev \
        libffi-dev \
        openssl \
        libssl-dev \
        tzdata \
        sqlite3 \
        libsqlite3-dev \
        libcurl4-openssl-dev \
        libexpat1-dev \
        uuid-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-dev \
        gcc \
        musl-dev \
        rsync

# Копируем код проекта в контейнер
WORKDIR /app

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "predict.py"]

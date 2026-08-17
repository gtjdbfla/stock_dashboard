FROM python:3.11-slim

ENV TZ=Asia/Seoul
RUN apt-get update && apt-get install -y --no-install-recommends tzdata fonts-nanum && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY over_market.py .
COPY collector.py .
COPY .streamlit/ .streamlit/

EXPOSE 8501

# 같은 이미지를 대시보드와 시간외 수집기가 함께 쓴다.
# 수집기는 docker-compose에서 command로 collector.py를 지정해 띄운다.
CMD ["streamlit", "run", "app.py"]

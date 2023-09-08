FROM python:alpine3.17
WORKDIR /app
COPY ./requirements.txt .
RUN python3 -m venv venv
RUN . venv/bin/activate
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py"]

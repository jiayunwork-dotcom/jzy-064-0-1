FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY tests ./tests
COPY pytest.ini .

# Automated tests run as part of the image build: a failing suite fails
# the build, so an image that exists is one whose tests passed.
RUN python -m pytest tests -q

# Case archive lives inside the container filesystem.
ENV CASES_PATH=/data/cases.json
RUN mkdir -p /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

from fastapi import FastAPI

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI()


@app.get("/predict/{text}")
async def predict(text):
    # whatever 
    return {"sentiment": sentiment_pipeline([text])[0]["label"]}

@app.get("/")
async def root():
    return {"message": "Hello World"}

# python3 app.py
# uvicorn main:app --reload
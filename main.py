from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{text}")
async def predict(text):
    # whatever 
    return {"sentiment": sentiment_pipeline([text])[0]["label"]}

@app.get("/")
async def root():
    return {"message": "Hello World"}

# python3 app.py
# uvicorn main:app --reload
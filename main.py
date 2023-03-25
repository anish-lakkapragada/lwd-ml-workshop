from transformers import pipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS noqa

app = FastAPI()

sentiment_pipeline = pipeline("sentiment-analysis")

origins = ["*"]

# CORS noqa
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping_fn(): 
    return {"ping": "pong"}

@app.get("/predict/{text}")
def predict_fn(text): 
    # text is a string
    prediction = sentiment_pipeline([text])[0]
    return {"sentiment": prediction["label"]}
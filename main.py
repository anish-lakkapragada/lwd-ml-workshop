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
async def ping_fn(): 
    return {"ping": "pong"}

@app.get("/predict/{text}")
async def predict(text): 
    predictions = sentiment_pipeline([text])
    return {"prediction": predictions[0]["label"]}

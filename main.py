from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("code_language_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "code": ""})

@app.post("/predict_html", response_class=HTMLResponse)
def predict_html(request: Request, code: str = Form(...)):
    X = vectorizer.transform([code])
    pred = model.predict(X)
    lang = label_encoder.inverse_transform(pred)[0]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": lang,
        "code": code
    })

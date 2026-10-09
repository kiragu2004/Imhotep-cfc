# Imhotep CfC Hazard API
import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from imhotep_model import FEATURES, TARGETS, SEQ_LEN, ImhotepMultiCfC

COUNTY_MODELS = {"muranga": "imhotep_cfc_muranga.pth"}
app = FastAPI(title="Imhotep CfC Hazard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

class LoadedModel:
    def __init__(self, path, cal_path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model = ImhotepMultiCfC(input_size=len(FEATURES))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean = np.array(ckpt["mean"], dtype=np.float32)
        self.std = np.array(ckpt["std"], dtype=np.float32)
        cal = np.load(cal_path)
        self.calibrators = {}
        for name in TARGETS:
            self.calibrators[name] = (cal[name + "_x"], cal[name + "_y"])
    def predict(self, sequence):
        x = (np.array(sequence, dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.tensor(x).unsqueeze(0)).numpy().ravel()
        raw = 1 / (1 + np.exp(-logits))
        out = {}
        for i, name in enumerate(TARGETS):
            xs, ys = self.calibrators[name]
            out[name] = float(np.interp(raw[i], xs, ys))
        return out

MODELS = {}
@app.on_event("startup")
def _load():
    for key, fname in COUNTY_MODELS.items():
        if os.path.exists(fname) and os.path.exists("calibrators.npz"):
            MODELS[key] = LoadedModel(fname, "calibrators.npz")
            print("loaded: " + key)

class PredictRequest(BaseModel):
    sequence: list[list[float]]

@app.get("/", response_class=HTMLResponse)
def root():
    counties = sorted(MODELS.keys())
    rows = "".join("<li><code>" + c + "</code></li>" for c in counties) or "<li>none yet</li>"
    html = ("<html><head><title>Imhotep CfC API</title>"
            "<style>body{font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem;background:#0a0e14;color:#dde7f0}code{color:#00d9ff}a{color:#00d9ff}h1{color:#00d9ff}</style></head><body>"
            "<h1>Imhotep CfC API</h1><p>Decision-support hazard forecasts for Kenyan counties.</p>"
            "<p><b>Trained counties:</b></p><ul>" + rows + "</ul>"
            "<p><a href='/docs'>Interactive API docs (Swagger)</a></p>"
            "<p><a href='/health'>Health check</a></p>"
            "<p style='color:#7a8896;font-size:.85rem;margin-top:2rem'>Decision support only. Not an official warning. Consult KMD, ICPAC, or county offices.</p></body></html>")
    return html

@app.get("/health")
def health():
    return {"status": "ok", "counties_trained": sorted(MODELS.keys()), "targets": TARGETS, "features": FEATURES, "seq_len": SEQ_LEN, "note": "Decision support only. Not an official warning."}

@app.get("/counties")
def counties():
    return {"counties": sorted(MODELS.keys()), "targets": TARGETS}

@app.post("/predict/{county}")
def predict(county: str, req: PredictRequest):
    key = county.lower().replace("'", "").replace(" ", "_")
    if key not in MODELS:
        raise HTTPException(404, "No trained model for '" + county + "'. Available: " + str(sorted(MODELS.keys())))
    x = np.array(req.sequence, dtype=np.float32)
    if x.shape != (SEQ_LEN, len(FEATURES)):
        raise HTTPException(400, "Expected " + str(SEQ_LEN) + "x" + str(len(FEATURES)) + ", got " + str(tuple(x.shape)))
    probs = MODELS[key].predict(req.sequence)
    return {"county": county, "predictions": probs, "note": "Decision support only. Not an official warning. Consult KMD, ICPAC, or county offices."}

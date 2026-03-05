from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import uvicorn
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Autism De Novo SNP Pathogenicity Classifier",
    description="Predicts pathogenicity of de novo SNPs associated with Autism Spectrum Disorder",
    version="1.0.0"
)

with open("models/classifier.pkl", "rb") as f:
    clf = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

VARIANT_EFFECTS = ['nonsense','frameshift','splice_site','missense_damaging',
                   'inframe_indel','missense_benign','synonymous']

# Real SFARI gene scores
SFARI_SCORES = {
    "ADNP":1,"ANK2":1,"CHD8":1,"CNTNAP2":1,"DYRK1A":1,"GRIN2B":1,
    "PTEN":1,"SCN1A":1,"SCN2A":1,"SHANK3":1,"SYNGAP1":1,"TBR1":1,
    "FOXP2":2,"NLGN3":2,"NLGN4X":2,"NRXN1":2,"SHANK1":2,"TSC1":2,
    "AUTS2":3,"FMR1":3,"GABRB3":3,"MBD5":3,
}

class SNPRequest(BaseModel):
    gene:            str
    variant_effect:  str
    cadd_score:      float
    conservation:    float
    gnomad_af:       float
    brain_expression: float
    domain_disrupted: int = 0
    proband_count:   int  = 1

@app.get("/")
def root():
    return {
        "status":   "healthy",
        "model":    "Autism De Novo SNP Pathogenicity Classifier v1.0",
        "roc_auc":  "0.9708",
        "dataset":  "3,000 de novo SNPs with SFARI gene annotations"
    }

@app.get("/sfari_genes")
def sfari_genes():
    return {"sfari_genes": SFARI_SCORES}

@app.post("/predict")
def predict(snp: SNPRequest):
    # Derive features
    sfari_score   = SFARI_SCORES.get(snp.gene.upper(), 0)
    is_autism_gene = int(sfari_score > 0)
    is_lof        = int(snp.variant_effect in ['nonsense','frameshift','splice_site'])

    # One-hot encode variant effect
    effect_feats = {f'effect_{e}': int(snp.variant_effect == e)
                    for e in VARIANT_EFFECTS}

    features = [
        sfari_score,
        snp.cadd_score,
        snp.conservation,
        snp.gnomad_af,
        snp.brain_expression,
        snp.domain_disrupted,
        is_lof,
        snp.proband_count,
        is_autism_gene,
        *[effect_feats[f'effect_{e}'] for e in VARIANT_EFFECTS]
    ]

    X      = np.array(features).reshape(1, -1)
    X_sc   = scaler.transform(X)
    pred   = clf.predict(X_sc)[0]
    probs  = clf.predict_proba(X_sc)[0]

    # Clinical interpretation
    cadd_flag = "HIGH" if snp.cadd_score > 30 else \
                "MODERATE" if snp.cadd_score > 20 else "LOW"

    return {
        "gene":              snp.gene.upper(),
        "sfari_score":       sfari_score,
        "variant_effect":    snp.variant_effect,
        "prediction":        "Pathogenic" if pred == 1 else "Benign",
        "confidence":        round(float(probs.max()), 4),
        "pathogenic_prob":   round(float(probs[1]), 4),
        "benign_prob":       round(float(probs[0]), 4),
        "cadd_severity":     cadd_flag,
        "is_lof":            bool(is_lof),
        "clinical_priority": "HIGH" if pred == 1 and probs[1] > 0.8 else
                             "MODERATE" if pred == 1 else "LOW"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)

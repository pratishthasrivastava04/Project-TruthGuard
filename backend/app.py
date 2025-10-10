# app.py


import os
import re
import math
import json
import tldextract
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Optional libraries (installed via requirements.txt)
try:
    from newspaper import Article
except Exception:
    Article = None

try:
    from langdetect import detect
except Exception:
    detect = None

# Optional: for better ML classification (install transformers + torch)
USE_TRANSFORMER_IF_AVAILABLE = True  # set True below if you want automatic attempt

transformer_pipeline = None
transformer_pipeline = None
try:
    if USE_TRANSFORMER_IF_AVAILABLE:
        from transformers import pipeline
        transformer_pipeline = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

        print("[INFO] Transformer model loaded successfully")
except Exception as e:
    transformer_pipeline = None
    print(f"[ERROR] Could not load transformer model: {e}")

load_dotenv()

# Optional Google Fact Check API key (put in .env as FACT_CHECK_API_KEY)
FACT_CHECK_API_KEY = os.getenv("FACT_CHECK_API_KEY", "").strip()

app = Flask(__name__)
CORS(app)  # allow requests from your frontend (localhost)

# Basic domain reputation lists (starter examples)
# You should expand these lists in production with curated sources.
REPUTABLE_DOMAINS = {
    "nytimes.com", "bbc.co.uk", "bbc.com", "theguardian.com", "reuters.com",
    "apnews.com", "washingtonpost.com", "wsj.com"
}
SUSPECT_DOMAINS = {
    "yourclickbaitexample.com", "fakenews-example.org", "clickbait.example"
}

# Helper functions
def extract_text_from_url(url):
    """Try to extract article text using newspaper3k if available; otherwise simple fetch fallback."""
    if not Article:
        return None, "newspaper3k not installed"
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        return text, None
    except Exception as e:
        return None, str(e)

def domain_reputation_score(domain):
    """Return 0-100 score for domain reliability based on lists/heuristics."""
    domain = domain.lower()
    if domain in REPUTABLE_DOMAINS:
        return 95
    if domain in SUSPECT_DOMAINS:
        return 10
    # Heuristic: longer TLDs or odd subdomain patterns reduce score
    score = 65
    if domain.endswith(".co") or domain.endswith(".info") or domain.endswith(".cc"):
        score -= 15
    if re.search(r"[0-9]", domain):
        score -= 10
    return max(10, min(95, score))

def uppercase_ratio(text):
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    up = sum(1 for c in letters if c.isupper())
    return up / len(letters)

def punctuation_scores(text):
    excls = text.count("!")
    qmarks = text.count("?")
    ellipses = text.count("...")
    return excls, qmarks, ellipses

def compute_language_score(text):
    """Language-based heuristics: grammar-like proxies (caps, punctuation, sensational words)."""
    score = 70
    uratio = uppercase_ratio(text)
    if uratio > 0.12:
        score -= 25
    excls, qmarks, ellipses = punctuation_scores(text)
    if excls > 2:
        score -= min(20, excls * 2)
    # sensational words penalty
    sensational = ["shocking", "you won't believe", "miracle", "sensational", "exposed", "allegedly"]
    lowered = text.lower()
    for w in sensational:
        if w in lowered:
            score -= 10
    # short text means less evidence
    if len(text) < 120:
        score -= 10
    return max(5, min(95, score))

def simple_fact_check_claims(text):
    """Optional quick heuristics: many claims with numbers or named people increases chance of check-needed"""
    numbers = len(re.findall(r"\b\d{2,}\b", text)) + len(re.findall(r"\b\d+%|\d+ percent\b", text))
    capitalized_phrases = len(re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3}\b", text))
    score = 50
    score -= min(25, numbers * 5)
    return max(5, min(95, score))

def call_google_factcheck(query):
    """Call Google Fact Check Tools API if API key is configured.
       Returns list of matches (may be empty)."""
    if not FACT_CHECK_API_KEY:
        return {"error":"no_api_key"}
    import requests
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": FACT_CHECK_API_KEY, "query": query}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def build_final_verdict(scores):
    """Blend component scores into final label and confidence."""
    # Weighted combination
    w_source = 0.35
    w_language = 0.25
    w_fact = 0.25
    w_claim = 0.15
    combined = (scores["source"] * w_source +
                scores["language"] * w_language +
                scores["fact"] * w_fact +
                scores["claim"] * w_claim)
    # Combined is reliability-like: higher = more credible
    credibility = combined
    # Map to label
    if credibility >= 70:
        label = "Likely Credible"
    elif credibility >= 45:
        label = "Uncertain"
    else:
        label = "Likely Misinformation"
    # Confidence: how far from 50
    confidence = int(min(98, max(10, abs(credibility - 50) + 50)))
    return label, confidence

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    url = (data.get("url") or "").strip()
    if not text and not url:
        return jsonify({"error":"Provide either text or url in JSON body."}), 400

    source_domain = None
    extracted_text = None
    extraction_message = None

    if url and not text:
        # extract text
        extracted_text, extraction_message = extract_text_from_url(url)
        if extracted_text:
            text = extracted_text
        else:
            # fallback: we will try to use the URL's path and domain as context
            text = text or ""
    if url:
        try:
            parsed = urlparse(url)
            tld = tldextract.extract(url)
            domain = ".".join(part for part in (tld.domain, tld.suffix) if part)
            source_domain = domain
        except Exception:
            source_domain = parsed.netloc if 'parsed' in locals() else None

    # Component scoring
    source_score = 50
    if source_domain:
        source_score = domain_reputation_score(source_domain)
    else:
        # If no domain, weak prior
        source_score = 50

    language_score = compute_language_score(text)
    claim_score = simple_fact_check_claims(text)  # lower is worse (we'll treat as "fact-check need")
    # Convert claim_score to "fact" friendly orientation (higher = more credible => invert)
    # we defined claim_score such that higher = less suspicious, so keep as-is.

    # Optional: call Google Fact Check API for matches
    fact_matches = None
    fact_api_info = {}
    fact_score = 60
    if FACT_CHECK_API_KEY:
        fc = call_google_factcheck(text[:300])  # send short query
        if isinstance(fc, dict) and fc.get("error"):
            fact_api_info = {"error": fc.get("error")}
        else:
            # if results found, increase fact_score if reliable sources confirm or decrease if debunked
            matches = fc.get("claims", []) if isinstance(fc, dict) else []
            fact_matches = matches
            if matches and len(matches) > 0:
                # if any claim has claimant and textual rating: heuristics
                found_false = False
                found_true = False
                for c in matches:
                    # Google fact-check claim structure can vary; try to find textual verdict fields
                    review = (c.get("claimReview") or [])
                    for r in review:
                        title = r.get("title","").lower()
                        if "false" in title or "misleading" in title or "pants on fire" in title:
                            found_false = True
                        if "true" in title or "true:" in title or "verified" in title:
                            found_true = True
                if found_false and not found_true:
                    fact_score = 15
                elif found_true and not found_false:
                    fact_score = 90
                else:
                    fact_score = 55
            else:
                fact_score = 55
    else:
        # No API key: estimate fact_score from content
        fact_score = max(10, 85 - (len(re.findall(r"\b(unverified|rumor|alleged|claim|report)\b", text.lower())) * 15))

    scores = {
        "source": int(source_score),
        "language": int(language_score),
        "fact": int(fact_score),
        "claim": int(claim_score)
    }

    label, confidence = build_final_verdict(scores)

    # Optional transformer classification (if available)
    transformer_result = None
    if transformer_pipeline:
        try:
            out = transformer_pipeline(text[:512])  # short chunk
            transformer_result = out
            # If model label suggests 'fake' adjust
            # NOTE: labels depend on the pipeline/model
        except Exception as e:
            transformer_result = {"error": str(e)}

    response = {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "source_domain": source_domain,
        "extracted_text_snippet": text[:1000] if text else None,
        "fact_matches": fact_matches,
        "fact_api_info": fact_api_info,
        "transformer_result": transformer_result,
    }
    return jsonify(response), 200

@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({"ok": True, "message": "TruthGuard backend running"}), 200

if __name__ == "__main__":
    print("[DEBUG] __main__ block running...")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    print(f"[DEBUG] Running Flask on port {port}, debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)

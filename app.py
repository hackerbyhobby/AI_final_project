import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import shap
import requests
import json
import os
import numpy as np

# Translator instance
translator = GoogleTranslator(source="auto", target="es")

# 1. Load separate keywords for SMiShing and Other Scam (assumed in English)
with open("smishing_keywords.txt", "r", encoding="utf-8") as f:
    SMISHING_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

with open("other_scam_keywords.txt", "r", encoding="utf-8") as f:
    OTHER_SCAM_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

# 2. Zero-Shot Classification Pipeline
model_name = "joeddav/xlm-roberta-large-xnli"
classifier = pipeline("zero-shot-classification", model=model_name)
CANDIDATE_LABELS = ["SMiShing", "Other Scam", "Legitimate"]

# SHAP explainer setup
explainer = shap.Explainer(classifier)

# Retrieve the Google Safe Browsing API key from the environment
SAFE_BROWSING_API_KEY = os.getenv("SAFE_BROWSING_API_KEY")

if not SAFE_BROWSING_API_KEY:
    raise ValueError("Google Safe Browsing API key not found. Please set it as an environment variable in your Hugging Face Space.")

SAFE_BROWSING_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

def check_url_with_google_safebrowsing(url):
    """
    Check a URL against Google's Safe Browsing API.
    """
    payload = {
        "client": {
            "clientId": "your-client-id",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [
                {"url": url}
            ]
        }
    }
    try:
        response = requests.post(
            SAFE_BROWSING_URL,
            params={"key": SAFE_BROWSING_API_KEY},
            json=payload
        )
        response_data = response.json()
        if "matches" in response_data:
            return True  # URL is flagged as malicious
        return False  # URL is safe
    except Exception as e:
        print(f"Error checking URL with Safe Browsing API: {e}")
        return False

def get_keywords_by_language(text: str):
    """
    Detect language using `langdetect` and translate keywords if needed.
    """
    snippet = text[:200]  # Use a snippet for detection
    try:
        detected_lang = detect(snippet)
    except Exception:
        detected_lang = "en"  # Default to English if detection fails

    if detected_lang == "es":
        smishing_in_spanish = [
            translator.translate(kw).lower() for kw in SMISHING_KEYWORDS
        ]
        other_scam_in_spanish = [
            translator.translate(kw).lower() for kw in OTHER_SCAM_KEYWORDS
        ]
        return smishing_in_spanish, other_scam_in_spanish, "es"
    else:
        return SMISHING_KEYWORDS, OTHER_SCAM_KEYWORDS, "en"

def boost_probabilities(probabilities: dict, text: str):
    """
    Boost probabilities based on keyword matches and presence of URLs.
    """
    lower_text = text.lower()
    smishing_keywords, other_scam_keywords, detected_lang = get_keywords_by_language(text)

    smishing_count = sum(1 for kw in smishing_keywords if kw in lower_text)
    other_scam_count = sum(1 for kw in other_scam_keywords if kw in lower_text)

    smishing_boost = 0.30 * smishing_count
    other_scam_boost = 0.30 * other_scam_count

    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    if found_urls:
        smishing_boost += 0.35

    p_smishing = probabilities.get("SMiShing", 0.0)
    p_other_scam = probabilities.get("Other Scam", 0.0)
    p_legit = probabilities.get("Legitimate", 1.0)

    p_smishing += smishing_boost
    p_other_scam += other_scam_boost
    p_legit -= (smishing_boost + other_scam_boost)

    p_smishing = max(p_smishing, 0.0)
    p_other_scam = max(p_other_scam, 0.0)
    p_legit = max(p_legit, 0.0)

    total = p_smishing + p_other_scam + p_legit
    if total > 0:
        p_smishing /= total
        p_other_scam /= total
        p_legit /= total
    else:
        p_smishing, p_other_scam, p_legit = 0.0, 0.0, 1.0

    return {
        "SMiShing": p_smishing,
        "Other Scam": p_other_scam,
        "Legitimate": p_legit,
        "detected_lang": detected_lang,
    }

def explain_classification(text):
    """
    Generate SHAP explanations for the classification.
    """
    shap_values = explainer([text])
    shap.force_plot(
        explainer.expected_value[0], shap_values[0].values[0], shap_values[0].data
    )

def smishing_detector(text, image):
    """
    Main detection function combining text and OCR.
    """
    combined_text = text or ""
    if image is not None:
        ocr_text = pytesseract.image_to_string(image, lang="spa+eng")
        combined_text += " " + ocr_text
    combined_text = combined_text.strip()

    if not combined_text:
        return {
            "text_used_for_classification": "(none)",
            "label": "No text provided",
            "confidence": 0.0,
            "keywords_found": [],
            "urls_found": [],
            "threat_analysis": "No URLs to analyze",
        }

    result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = {k: float(v) for k, v in zip(result["labels"], result["scores"])}
    boosted = boost_probabilities(original_probs, combined_text)

    boosted = {k: float(v) for k, v in boosted.items() if isinstance(v, (int, float))}
    detected_lang = boosted.pop("detected_lang", "en")
    final_label = max(boosted, key=boosted.get)
    final_confidence = round(boosted[final_label], 3)

    lower_text = combined_text.lower()
    smishing_keys, scam_keys, _ = get_keywords_by_language(combined_text)

    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    found_smishing = [kw for kw in smishing_keys if kw in lower_text]
    found_other_scam = [kw for kw in scam_keys if kw in lower_text]

    # Analyze URLs using Google's Safe Browsing API
    threat_analysis = {
        url: check_url_with_google_safebrowsing(url) for url in found_urls
    }

    # SHAP Explanation (optional for user insights)
    explain_classification(combined_text)

    return {
        "detected_language": detected_lang,
        "text_used_for_classification": combined_text,
        "original_probabilities": {k: round(v, 3) for k, v in original_probs.items()},
        "boosted_probabilities": {k: round(v, 3) for k, v in boosted.items()},
        "label": final_label,
        "confidence": final_confidence,
        "smishing_keywords_found": found_smishing,
        "other_scam_keywords_found": found_other_scam,
        "urls_found": found_urls,
        "threat_analysis": threat_analysis,
    }

demo = gr.Interface(
    fn=smishing_detector,
    inputs=[
        gr.Textbox(
            lines=3,
            label="Paste Suspicious SMS Text (English/Spanish)",
            placeholder="Type or paste the message here..."
        ),
        gr.Image(
            type="pil",
            label="Or Upload a Screenshot (Optional)"
        )
    ],
    outputs="json",
    title="SMiShing & Scam Detector with Safe Browsing",
    description="""
This tool classifies messages as SMiShing, Other Scam, or Legitimate using a zero-shot model
(joeddav/xlm-roberta-large-xnli). It automatically detects if the text is Spanish or English.
It uses SHAP for explainability and checks URLs against Google's Safe Browsing API for enhanced analysis.
    """,
    flagging_mode="true"
)

if __name__ == "__main__":
    demo.launch()
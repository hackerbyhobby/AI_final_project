import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re
from langdetect import detect
from deep_translator import GoogleTranslator

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

def get_keywords_by_language(text: str):
    """
    1. Detect language using `langdetect`.
    2. If Spanish ('es'), translate each English-based keyword to Spanish using `deep-translator`.
    3. If English (or other languages), use the original English lists.
    """
    snippet = text[:200]  # Use a snippet for detection
    try:
        detected_lang = detect(snippet)
    except Exception:
        detected_lang = "en"  # Default to English if detection fails

    if detected_lang == "es":
        # Translate all SMiShing and Other Scam keywords to Spanish
        smishing_in_spanish = [
            translator.translate(kw).lower() for kw in SMISHING_KEYWORDS
        ]
        other_scam_in_spanish = [
            translator.translate(kw).lower() for kw in OTHER_SCAM_KEYWORDS
        ]
        return smishing_in_spanish, other_scam_in_spanish, "es"
    else:
        # Default to English keywords
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

    p_smishing = probabilities["SMiShing"]
    p_other_scam = probabilities["Other Scam"]
    p_legit = probabilities["Legitimate"]

    p_smishing += smishing_boost
    p_other_scam += other_scam_boost
    p_legit -= (smishing_boost + other_scam_boost)

    if p_smishing < 0:
        p_smishing = 0.0
    if p_other_scam < 0:
        p_other_scam = 0.0
    if p_legit < 0:
        p_legit = 0.0

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
        "detected_lang": detected_lang
    }

def smishing_detector(text, image):
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
            "urls_found": []
        }

    result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = dict(zip(result["labels"], result["scores"]))
    boosted = boost_probabilities(original_probs, combined_text)
    final_label = max(boosted, key=boosted.get)
    final_confidence = round(boosted[final_label], 3)
    detected_lang = boosted.pop("detected_lang", "en")

    lower_text = combined_text.lower()
    smishing_keys, scam_keys, _ = get_keywords_by_language(combined_text)

    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    found_smishing = [kw for kw in smishing_keys if kw in lower_text]
    found_other_scam = [kw for kw in scam_keys if kw in lower_text]

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
    title="SMiShing & Scam Detector (Language Detection + Keyword Translation)",
    description="""
This tool classifies messages as SMiShing, Other Scam, or Legitimate using a zero-shot model
(joeddav/xlm-roberta-large-xnli). It automatically detects if the text is Spanish or English.
If Spanish, it translates the English-based keyword lists to Spanish before boosting the scores.
Any URL found further boosts SMiShing specifically.
""",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re

# Language detection & translation
from langdetect import detect
from googletrans import Translator

translator = Translator()

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
    1. Detect language (using `langdetect`).
    2. If Spanish ('es'), translate each English-based keyword to Spanish using googletrans.
    3. If English (or anything else), just use the original English lists.
    """
    # Attempt to detect language from a snippet (to reduce overhead on very large text)
    snippet = text[:200]  # up to 200 chars for detection
    try:
        detected_lang = detect(snippet)
    except:
        detected_lang = "en"  # fallback if detection fails

    if detected_lang == "es":
        # Translate all SMiShing and Other Scam keywords to Spanish
        smishing_in_spanish = [
            translator.translate(kw, src="en", dest="es").text.lower()
            for kw in SMISHING_KEYWORDS
        ]
        other_scam_in_spanish = [
            translator.translate(kw, src="en", dest="es").text.lower()
            for kw in OTHER_SCAM_KEYWORDS
        ]
        return smishing_in_spanish, other_scam_in_spanish, "es"
    else:
        # Default to English keywords
        return SMISHING_KEYWORDS, OTHER_SCAM_KEYWORDS, "en"

def boost_probabilities(probabilities: dict, text: str):
    """
    1. Load the appropriate keyword lists (English or Spanish).
    2. Count matches for SMiShing vs. Other Scam.
    3. If a URL is found, add an extra boost only to SMiShing.
    4. Subtract total boost from 'Legitimate'.
    5. Clamp negative probabilities to 0, re-normalize.
    """
    lower_text = text.lower()

    # Grab the correct keyword lists based on language
    smishing_keywords, other_scam_keywords, detected_lang = get_keywords_by_language(text)

    # Count SMiShing keyword matches
    smishing_count = sum(1 for kw in smishing_keywords if kw in lower_text)
    # Count Other Scam keyword matches
    other_scam_count = sum(1 for kw in other_scam_keywords if kw in lower_text)

    # Base boost amounts
    smishing_boost = 0.30 * smishing_count
    other_scam_boost = 0.30 * other_scam_count

    # Check for URLs => +0.35 only to SMiShing
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    if found_urls:
        smishing_boost += 0.35

    # Extract original probabilities
    p_smishing = probabilities["SMiShing"]
    p_other_scam = probabilities["Other Scam"]
    p_legit = probabilities["Legitimate"]

    # Apply boosts
    p_smishing += smishing_boost
    p_other_scam += other_scam_boost

    # Subtract total boost from 'Legitimate'
    total_boost = smishing_boost + other_scam_boost
    p_legit -= total_boost

    # Clamp negative probabilities
    if p_smishing < 0:
        p_smishing = 0.0
    if p_other_scam < 0:
        p_other_scam = 0.0
    if p_legit < 0:
        p_legit = 0.0

    # Re-normalize
    total = p_smishing + p_other_scam + p_legit
    if total > 0:
        p_smishing /= total
        p_other_scam /= total
        p_legit /= total
    else:
        # fallback if everything is 0
        p_smishing, p_other_scam, p_legit = 0.0, 0.0, 1.0

    return {
        "SMiShing": p_smishing,
        "Other Scam": p_other_scam,
        "Legitimate": p_legit,
        "detected_lang": detected_lang
    }

def smishing_detector(text, image):
    """
    Main function called by Gradio.
    1. Combine user text + OCR text (if an image is provided).
    2. Zero-shot classify => base probabilities.
    3. Apply language detection & translation if needed, then boost logic.
    4. Return final classification.
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
            "urls_found": []
        }

    # 1. Zero-shot classification
    result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = dict(zip(result["labels"], result["scores"]))

    # 2. Boost logic (including language detection + translation)
    boosted = boost_probabilities(original_probs, combined_text)
    final_label = max(boosted, key=boosted.get) if not isinstance(boosted.get("detected_lang"), float) else "Legitimate"
    # to avoid conflict, let's store the detected language separately:
    detected_lang = boosted.pop("detected_lang", "en")

    # We have p_smishing, p_other_scam, p_legit left in boosted
    final_label = max(boosted, key=boosted.get)
    final_confidence = round(boosted[final_label], 3)

    # 3. Identify which keywords & URLs we found
    lower_text = combined_text.lower()
    # If we detected Spanish, we used the translated keywords to do matching. But let's also show them:
    # For demonstration, let's just show the "English or Spanish" keywords. The code to show them in output
    # can be the same as before, or you can do a second pass with the same logic from boost_probabilities.
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)

    # We'll do a quick second pass on actual matched keywords so user sees them
    # - If language is es => we used translated Spanish keywords, let's do the same for display
    # - If language is en => we used the original English lists
    if detected_lang == "es":
        smishing_keys, scam_keys, _ = get_keywords_by_language(combined_text)
    else:
        smishing_keys, scam_keys, _ = (SMISHING_KEYWORDS, OTHER_SCAM_KEYWORDS, "en")

    found_smishing = [kw for kw in smishing_keys if kw in lower_text]
    found_other_scam = [kw for kw in scam_keys if kw in lower_text]

    return {
        "detected_language": detected_lang,
        "text_used_for_classification": combined_text,
        "original_probabilities": {
            k: round(v, 3) for k, v in original_probs.items()
        },
        "boosted_probabilities": {
            k: round(v, 3) for k, v in boosted.items()
        },
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
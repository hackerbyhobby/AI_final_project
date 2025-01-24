import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re

# 1. Load scam keywords from file
#    Each line in 'scam_keywords.txt' is treated as a separate keyword.
with open("scam_keywords.txt", "r", encoding="utf-8") as f:
    SCAM_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

# 2. Zero-Shot Classification Pipeline
model_name = "joeddav/xlm-roberta-large-xnli"
classifier = pipeline("zero-shot-classification", model=model_name)

CANDIDATE_LABELS = ["SMiShing", "Other Scam", "Legitimate"]

def keyword_and_url_boost(probabilities, text):
    """
    Adjust final probabilities if certain scam-related keywords or URLs appear.
      - probabilities: dict, label -> original probability
      - text: the combined text from user input + OCR

    Returns an updated dict of probabilities that sum to 1.
    """
    lower_text = text.lower()

    # 1. Check scam keywords
    keyword_count = sum(1 for kw in SCAM_KEYWORDS if kw in lower_text)
    keyword_boost = 0.05 * keyword_count  # 5% per found keyword
    keyword_boost = min(keyword_boost, 0.30)  # cap at +30%

    # 2. Check if there's any URL (simple regex for http/https)
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    url_boost = 0.0
    if found_urls:
        # For demonstration: a flat +10% if a URL is found
        url_boost = 0.10

    # 3. Combine total boost
    total_boost = keyword_boost + url_boost
    total_boost = min(total_boost, 0.40)  # cap at +40%

    if total_boost <= 0:
        return probabilities  # no change if no keywords/URLs found

    smishing_prob = probabilities["SMiShing"]
    other_scam_prob = probabilities["Other Scam"]
    legit_prob = probabilities["Legitimate"]

    # 4. Distribute the total boost equally to "SMiShing" and "Other Scam"
    half_boost = total_boost / 2.0
    smishing_boosted = smishing_prob + half_boost
    other_scam_boosted = other_scam_prob + half_boost
    legit_boosted = legit_prob

    # 5. Re-normalize so they sum to 1
    total = smishing_boosted + other_scam_boosted + legit_boosted
    if total > 0:
        smishing_final = smishing_boosted / total
        other_scam_final = other_scam_boosted / total
        legit_final = legit_boosted / total
    else:
        smishing_final = 0.0
        other_scam_final = 0.0
        legit_final = 1.0

    return {
        "SMiShing": smishing_final,
        "Other Scam": other_scam_final,
        "Legitimate": legit_final
    }

def smishing_detector(text, image):
    """
    1. Extract text from the image (OCR) if provided.
    2. Combine with user-entered text.
    3. Zero-shot classification -> base probabilities.
    4. Keyword + URL boost -> adjusted probabilities.
    5. Return final label, confidence, etc.
    """
    # Step 1: OCR if there's an image
    combined_text = text if text else ""
    if image is not None:
        ocr_text = pytesseract.image_to_string(image, lang="spa+eng")
        combined_text += " " + ocr_text

    # Clean text
    combined_text = combined_text.strip()
    if not combined_text:
        return {
            "text_used_for_classification": "(none)",
            "label": "No text provided",
            "confidence": 0.0,
            "keywords_found": [],
            "urls_found": []
        }

    # Step 2: Zero-shot classification
    result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = dict(zip(result["labels"], result["scores"]))

    # Step 3: Keyword + URL boost
    boosted_probs = keyword_and_url_boost(original_probs, combined_text)

    # Step 4: Pick final label after boost
    final_label = max(boosted_probs, key=boosted_probs.get)
    final_confidence = round(boosted_probs[final_label], 3)

    # Step 5: Identify which keywords and URLs were found
    lower_text = combined_text.lower()
    found_keywords = [kw for kw in SCAM_KEYWORDS if kw in lower_text]
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)

    return {
        "text_used_for_classification": combined_text,
        "original_probabilities": {k: round(v, 3) for k, v in original_probs.items()},
        "boosted_probabilities": {k: round(v, 3) for k, v in boosted_probs.items()},
        "label": final_label,
        "confidence": final_confidence,
        "keywords_found": found_keywords,
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
    title="SMiShing & Scam Detector (Keyword + URL Boost)",
    description="""
This tool classifies messages as SMiShing, Other Scam, or Legitimate using a zero-shot model
(joeddav/xlm-roberta-large-xnli). It also checks for certain "scam keywords" (loaded from a file)
and any URLs, boosting the probability of a scam label if found.
Supports English & Spanish text (OCR included).
""",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
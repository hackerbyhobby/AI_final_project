import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re

# 1. Load keywords from separate files
with open("smishing_keywords.txt", "r", encoding="utf-8") as f:
    SMISHING_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

with open("other_scam_keywords.txt", "r", encoding="utf-8") as f:
    OTHER_SCAM_KEYWORDS = [line.strip().lower() for line in f if line.strip()]

# 2. Load the zero-shot classification pipeline
model_name = "joeddav/xlm-roberta-large-xnli"
classifier = pipeline("zero-shot-classification", model=model_name)

# We will classify among these three labels
CANDIDATE_LABELS = ["SMiShing", "Other Scam", "Legitimate"]

def boost_probabilities(probabilities: dict, text: str) -> dict:
    """
    Increases SMiShing probability if 'smishing_keywords' or URLs are found.
    Increases Other Scam probability if 'other_scam_keywords' are found.
    Reduces Legitimate by the total amount of these boosts.
    Then clamps negative probabilities to 0 and re-normalizes.
    """
    lower_text = text.lower()

    # Count smishing keywords
    smishing_keyword_count = sum(1 for kw in SMISHING_KEYWORDS if kw in lower_text)
    # Count other scam keywords
    other_scam_keyword_count = sum(1 for kw in OTHER_SCAM_KEYWORDS if kw in lower_text)

    # Base boosts
    smishing_boost = 0.10 * smishing_keyword_count
    other_scam_boost = 0.10 * other_scam_keyword_count

    # Check URLs => +0.20 only to Smishing
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)
    if found_urls:
        smishing_boost += 0.20

    # Extract original probabilities
    p_smishing = probabilities["SMiShing"]
    p_other_scam = probabilities["Other Scam"]
    p_legit = probabilities["Legitimate"]

    # Apply boosts
    p_smishing += smishing_boost
    p_other_scam += other_scam_boost

    # Subtract total boost from Legitimate
    total_boost = smishing_boost + other_scam_boost
    p_legit -= total_boost

    # Clamp negative probabilities
    if p_smishing < 0:
        p_smishing = 0.0
    if p_other_scam < 0:
        p_other_scam = 0.0
    if p_legit < 0:
        p_legit = 0.0

    # Re-normalize so sum=1
    total = p_smishing + p_other_scam + p_legit
    if total > 0:
        p_smishing /= total
        p_other_scam /= total
        p_legit /= total
    else:
        # fallback if everything is zero
        p_smishing, p_other_scam, p_legit = 0.0, 0.0, 1.0

    return {
        "SMiShing": p_smishing,
        "Other Scam": p_other_scam,
        "Legitimate": p_legit
    }

def smishing_detector(text, image):
    """
    1. OCR if image provided.
    2. Zero-shot classify => base probabilities.
    3. Boost probabilities based on keywords + URL logic.
    4. Return final classification + confidence.
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
            "smishing_keywords_found": [],
            "other_scam_keywords_found": [],
            "urls_found": []
        }

    # Perform zero-shot classification
    result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = dict(zip(result["labels"], result["scores"]))

    # Apply boosts
    boosted_probs = boost_probabilities(original_probs, combined_text)
    final_label = max(boosted_probs, key=boosted_probs.get)
    final_confidence = round(boosted_probs[final_label], 3)

    # For display: which keywords + URLs
    lower_text = combined_text.lower()
    smishing_found = [kw for kw in SMISHING_KEYWORDS if kw in lower_text]
    other_scam_found = [kw for kw in OTHER_SCAM_KEYWORDS if kw in lower_text]
    found_urls = re.findall(r"(https?://[^\s]+)", lower_text)

    return {
        "text_used_for_classification": combined_text,
        "original_probabilities": {
            k: round(v, 3) for k, v in original_probs.items()
        },
        "boosted_probabilities": {
            k: round(v, 3) for k, v in boosted_probs.items()
        },
        "label": final_label,
        "confidence": final_confidence,
        "smishing_keywords_found": smishing_found,
        "other_scam_keywords_found": other_scam_found,
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
    title="SMiShing & Scam Detector (Separate Keywords + URL → SMiShing)",
    description="""
This tool classifies messages as SMiShing, Other Scam, or Legitimate using a zero-shot model
(joeddav/xlm-roberta-large-xnli). 
- 'smishing_keywords.txt' boosts SMiShing specifically.
- 'other_scam_keywords.txt' boosts Other Scam specifically.
- Any URL found further boosts ONLY Smishing.
- The total boost is subtracted from Legitimate.
Supports English & Spanish text (OCR included).
""",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
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
    Detect language using `langdetect` and translate keywords if needed.
    """
    snippet = text[:200]
    try:
        detected_lang = detect(snippet)
    except Exception:
        detected_lang = "en"

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

    # Clamp
    p_smishing = max(p_smishing, 0.0)
    p_other_scam = max(p_other_scam, 0.0)
    p_legit = max(p_legit, 0.0)

    # Re-normalize
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

def smishing_detector(input_type, text, image):
    """
    Main detection function combining text (if 'Text') and OCR (if 'Screenshot').
    """
    if input_type == "Text":
        combined_text = text.strip() if text else ""
    else:
        combined_text = ""
        if image is not None:
            combined_text = pytesseract.image_to_string(image, lang="spa+eng").strip()

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
    original_probs = {k: float(v) for k, v in zip(result["labels"], result["scores"])}

    boosted = boost_probabilities(original_probs, combined_text)

    # Patched snippet begins
    # 1. Extract language first, preserving it
    detected_lang = boosted.get("detected_lang", "en")
    # 2. Remove it so only numeric keys remain
    boosted.pop("detected_lang", None)
    # 3. Convert numeric values to float
    for k, v in boosted.items():
        boosted[k] = float(v)
    # Patched snippet ends

    final_label = max(boosted, key=boosted.get)
    final_confidence = round(boosted[final_label], 3)

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

#
# Gradio interface with dynamic visibility
#
def toggle_inputs(choice):
    """
    Return updates for (text_input, image_input) based on the radio selection.
    """
    if choice == "Text":
        # Show text input, hide image
        return gr.update(visible=True), gr.update(visible=False)
    else:
        # choice == "Screenshot"
        # Hide text input, show image
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks() as demo:
    gr.Markdown("## SMiShing & Scam Detector (Choose Text or Screenshot)")
    
    with gr.Row():
        input_type = gr.Radio(
            choices=["Text", "Screenshot"], 
            value="Text", 
            label="Choose Input Type"
        )

    text_input = gr.Textbox(
        lines=3,
        label="Paste Suspicious SMS Text",
        placeholder="Type or paste the message here...",
        visible=True  # default
    )

    image_input = gr.Image(
        type="pil",
        label="Upload Screenshot",
        visible=False  # hidden by default
    )

    # Whenever input_type changes, toggle which input is visible
    input_type.change(
        fn=toggle_inputs,
        inputs=input_type,
        outputs=[text_input, image_input],
        queue=False
    )

    # Button to run classification
    analyze_btn = gr.Button("Classify")
    output_json = gr.JSON(label="Result")

    # On button click, call the smishing_detector
    analyze_btn.click(
        fn=smishing_detector,
        inputs=[input_type, text_input, image_input],
        outputs=output_json
    )

if __name__ == "__main__":
    demo.launch()
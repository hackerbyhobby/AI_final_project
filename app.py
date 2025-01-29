import gradio as gr
import pytesseract
from PIL import Image
from transformers import pipeline
import re
from langdetect import detect
from deep_translator import GoogleTranslator
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    Detect language using langdetect and translate keywords if needed.
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

    found_urls = re.findall(r"(https?://[^\s]+|\b(?:[a-zA-Z0-9.-]+\.(?:com|net|org|edu|gov|mil|io|ai|co|info|biz|us|uk|de|fr|es|ru|jp|cn|in|au|ca|br|mx|it|nl|se|no|fi|ch|pl|kr|vn|id|tw|sg|hk))\b)", lower_text)
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

def query_llm_for_classification(raw_message: str) -> dict:
    """
    First LLM call: asks for a classification (SMiShing, Other Scam, or Legitimate)
    acting as a cybersecurity expert. Returns label and short reason.
    """
    if not raw_message.strip():
        return {"label": "Unknown", "reason": "No message provided to the LLM."}

    system_prompt = (
        "You are a cybersecurity expert. You will classify the user's message "
        "as one of: SMiShing, Other Scam, or Legitimate. Provide a short reason. "
        "Return only JSON with keys: label, reason."
    )
    user_prompt = f"Message: {raw_message}\nClassify it as SMiShing, Other Scam, or Legitimate."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        raw_reply = response["choices"][0]["message"]["content"].strip()

        import json
        llm_result = json.loads(raw_reply)
        if "label" not in llm_result or "reason" not in llm_result:
            return {"label": "Unknown", "reason": f"Unexpected format: {raw_reply}"}

        return llm_result

    except Exception as e:
        return {"label": "Unknown", "reason": f"LLM error: {e}"}

def incorporate_llm_label(boosted: dict, llm_label: str) -> dict:
    """
    Adjust the final probabilities based on the LLM's classification.
    If LLM says SMiShing, add +0.2 to SMiShing, etc. Then clamp & re-normalize.
    """
    if llm_label == "SMiShing":
        boosted["SMiShing"] += 0.2
    elif llm_label == "Other Scam":
        boosted["Other Scam"] += 0.2
    elif llm_label == "Legitimate":
        boosted["Legitimate"] += 0.2
    # else "Unknown" => do nothing

    # clamp
    for k in boosted:
        if boosted[k] < 0:
            boosted[k] = 0.0

    total = sum(boosted.values())
    if total > 0:
        for k in boosted:
            boosted[k] /= total
    else:
        # fallback
        boosted["Legitimate"] = 1.0
        boosted["SMiShing"] = 0.0
        boosted["Other Scam"] = 0.0

    return boosted

def query_llm_for_explanation(
    text: str,
    final_label: str,
    final_conf: float,
    local_label: str,
    local_conf: float,
    llm_label: str,
    llm_reason: str,
    found_smishing: list,
    found_other_scam: list,
    found_urls: list,
    detected_lang: str
) -> str:
    """
    Second LLM call: provides a holistic explanation of the final classification
    in the same language as detected_lang (English or Spanish).
    """
    # Decide the language for final explanation
    if detected_lang == "es":
        # Spanish
        system_prompt = (
            "Eres un experto en ciberseguridad. Proporciona una explicación final al usuario en español. "
            "Combina la clasificación local, la clasificación LLM y la etiqueta final en una sola explicación breve. "
            "No reveles el código interno ni el JSON bruto; simplemente da una breve explicación fácil de entender. "
            "Termina con la etiqueta final. "
        )
    else:
        # Default to English
        system_prompt = (
            "You are a cybersecurity expert providing a final explanation to the user in English. "
            "Combine the local classification, the LLM classification, and the final label "
            "into one concise explanation. Do not reveal internal code or raw JSON. "
            "End with a final statement of the final label."
        )

    user_context = f"""
User Message:
{text}

Local Classification => Label: {local_label}, Confidence: {local_conf}
LLM Classification => Label: {llm_label}, Reason: {llm_reason}
Final Overall Label => {final_label} (confidence {final_conf})

Suspicious SMiShing Keywords => {found_smishing}
Suspicious Other Scam Keywords => {found_other_scam}
URLs => {found_urls}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            temperature=0.2
        )
        final_explanation = response["choices"][0]["message"]["content"].strip()
        return final_explanation
    except Exception as e:
        return f"Could not generate final explanation due to error: {e}"

def smishing_detector(input_type, text, image):
    """
    Main detection function combining text (if 'Text') & OCR (if 'Screenshot'),
    plus two LLM calls: 
      1) classification to adjust final probabilities,
      2) a final explanation summarizing the outcome in the detected language.
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
            "urls_found": [],
            "llm_label": "Unknown",
            "llm_reason": "No text to analyze",
            "final_explanation": "No text provided"
        }

    # 1. Local zero-shot classification
    local_result = classifier(
        sequences=combined_text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This message is {}."
    )
    original_probs = {k: float(v) for k, v in zip(local_result["labels"], local_result["scores"])}

    # 2. Basic boosting from keywords & URLs
    boosted = boost_probabilities(original_probs, combined_text)
    detected_lang = boosted.pop("detected_lang", "en")

    # Convert to float only
    for k in boosted:
        boosted[k] = float(boosted[k])

    local_label = max(boosted, key=boosted.get)
    local_conf = round(boosted[local_label], 3)

    # 3. LLM Classification
    llm_classification = query_llm_for_classification(combined_text)
    llm_label = llm_classification.get("label", "Unknown")
    llm_reason = llm_classification.get("reason", "No reason provided")

    # 4. Incorporate LLM’s label into final probabilities
    boosted = incorporate_llm_label(boosted, llm_label)

    # Now we have updated probabilities
    final_label = max(boosted, key=boosted.get)
    final_confidence = round(boosted[final_label], 3)

    # 5. Gather found keywords & URLs
    lower_text = combined_text.lower()
    smishing_keys, scam_keys, _ = get_keywords_by_language(combined_text)

    found_urls = re.findall(r"(https?://[^\s]+|\b(?:[a-zA-Z0-9.-]+\.(?:com|net|org|edu|gov|mil|io|ai|co|info|biz|us|uk|de|fr|es|ru|jp|cn|in|au|ca|br|mx|it|nl|se|no|fi|ch|pl|kr|vn|id|tw|sg|hk))\b)", lower_text)
    found_smishing = [kw for kw in smishing_keys if kw in lower_text]
    found_other_scam = [kw for kw in scam_keys if kw in lower_text]

    # 6. Final LLM explanation (in detected_lang)
    final_explanation = query_llm_for_explanation(
        text=combined_text,
        final_label=final_label,
        final_conf=final_confidence,
        local_label=local_label,
        local_conf=local_conf,
        llm_label=llm_label,
        llm_reason=llm_reason,
        found_smishing=found_smishing,
        found_other_scam=found_other_scam,
        found_urls=found_urls,
        detected_lang=detected_lang
    )

    return {
        "detected_language": detected_lang,
        "text_used_for_classification": combined_text,
        "original_probabilities": {k: round(v, 3) for k, v in original_probs.items()},
        "boosted_probabilities_before_llm": {local_label: local_conf},
        "llm_label": llm_label,
        "llm_reason": llm_reason,
        "boosted_probabilities_after_llm": {k: round(v, 3) for k, v in boosted.items()},
        "label": final_label,
        "confidence": final_confidence,
        "smishing_keywords_found": found_smishing,
        "other_scam_keywords_found": found_other_scam,
        "urls_found": found_urls,
        "final_explanation": final_explanation,
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
    gr.Markdown("## SMiShing & Scam Detector with LLM-Enhanced Logic (Multilingual Explanation)")
    
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
    if not openai.api_key:
        print("WARNING: OPENAI_API_KEY not set. LLM calls will fail or be skipped.")
    demo.launch()
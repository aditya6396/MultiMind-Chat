import json
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def translate_text(text, tokenizer, model, src_lang="en_XX", tgt_lang="hi_IN"):
    # Tokenize the input text
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation using the model
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    
    # Decode the generated tokens to get the translated text
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def translate_json(json_data, tokenizer, model, src_lang="en_XX", tgt_lang="hi_IN"):
    if isinstance(json_data, dict):
        return {key: translate_json(value, tokenizer, model, src_lang, tgt_lang) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [translate_json(item, tokenizer, model, src_lang, tgt_lang) for item in json_data]
    elif isinstance(json_data, str):
        return translate_text(json_data, tokenizer, model, src_lang, tgt_lang)
    else:
        return json_data

def main():
    # Load the translation model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    # Load JSON content from file
    with open("json/q.json", "r", encoding="utf-8") as f:
        input_json = json.load(f)
    
    # Translate the JSON content to Hindi
    translated_json = translate_json(input_json, tokenizer, model)
    
    # Print the translated JSON with Devanagari script font
    # Since JSON itself does not control font styles, this is a placeholder for display
    print(json.dumps(translated_json, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    main()

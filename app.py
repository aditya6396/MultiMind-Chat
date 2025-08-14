from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr

model_name = "microsoft/BiomedVLP-BioViL-T"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def inference(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits.argmax().item()

iface = gr.Interface(fn=inference, inputs="text", outputs="text")
iface.launch()
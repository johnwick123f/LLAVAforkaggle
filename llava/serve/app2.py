import argparse
import torch
import gradio as gr
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
#from transformers import TextStreamer
model_name = get_model_name_from_path("/kaggle/working/LLaVA-7B-Lightening-v1-1")
#tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto")
tokenizer, model, image_processor, context_len = load_pretrained_model("/kaggle/working/LLaVA-7B-Lightening-v1-1", None, model_name, load_4bit=True)
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(image, query):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path("/kaggle/working/LLaVA-7B-Lightening-v1-1")
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
        
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = query
    print(f"{roles[0]}: {inp}")
    print(f"{roles[1]}: ", end="")
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    prompt = f"{roles[0]} {inp}\n{roles[1]}:"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#_ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
    #generation_kwargs = dict(input_ids, images=image_tensor, max_new_tokens=200, temperature=0.1, top_k=20, top_p=0.4, do_sample=True, repetition_penalty=1.2, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria])
    generation_kwargs = {
    "input_ids": input_ids,
    "images": image_tensor,
    "max_new_tokens": 200,
    "temperature": 0.1,
    "top_k": 20,
    "top_p": 0.4,
    "do_sample": True,
    "repetition_penalty": 1.2,
    "streamer": streamer,
    "use_cache": True,
    "stopping_criteria": [stopping_criteria]
    }

    #thread = Thread(target=model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,max_new_tokens=1024, streamer=streamer, use_cache=True, stopping_criteria=[stopping_criteria]))
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
        #print(new_text, end="", flush=True)
demo = gr.Interface(
    main, 
    inputs=[gr.Image(type="filepath"), "text"], 
    outputs="text",
    title="Llava demo",
    description="cool app for llava demo",
)
demo.queue()
demo.launch(debug=True)

import torch
from transformers import pipeline
import numpy as np
# from TTS.api import TTS
from playsound import playsound
import time
import os
import tkinter
os.environ["TOKENIZERS_PARALLELISM"] = "false"

classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# model_name = TTS.list_models()[0]
# tts = TTS(model_name) # optional parameters here in case of custom model.

output_dir = 'output'
model_type = 'gpt2'
model_name_or_path = 'microsoft/DialoGPT-large'
config_name = 'microsoft/DialoGPT-large'
tokenizer_name = 'microsoft/DialoGPT-large'
cache_dir = 'cached'

# chizuru_bot = torch.from_pretrained("path/to/model")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelWithLMHead
import torch
import transformers
from tkinter import *
from PIL import ImageTk, Image



transformers.logging.set_verbosity_error() # Prevent it from bitching about padding_side

def get_most_likely_emotion(emotion_array):
    flatten = emotion_array[0]
    chances = []
    for e in flatten:
        chances.append(e['score'])
    highest_score = np.argmax(chances)
    return flatten[highest_score]['label'] # Returns most-likely emotion

# def expression(emotion):
#     if emotion == "sadness":
#         return "(;_;)"
#     elif emotion == "joy":
#         return "\\(^-^)/"
#     elif emotion == "love":
#         return "(♥ω♥*)"
#     elif emotion == "anger":
#         return "(　ﾟДﾟ)＜!!"
#     elif emotion == "fear":
#         return "ヾ(ﾟдﾟ)ﾉ゛"
#     elif emotion == "surprise":
#         return "(⚆ᗝ⚆)"


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side = 'left')
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large") # Will load from file once it's trained.
config = AutoConfig.from_pretrained(config_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained( # LM head, recall it from Andrej Karpathy's nanoGPT tutorial. I think it either refers to how transformers/masking/token probabilities are calculated, OR the linear models that help bend inputs/outputs into certain shapes.
    'output'
    )

# device = torch.device("cuda")
# model.to(device)

def model_is_broken(output):
    if (output == ""): # This would be a machine learning model. Right now it's just a stub.
        return True
    return False

step = 0
chat_history_ids = None

def generate(input):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    global step
    global chat_history_ids
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # store history before input and output in a temp variable, incase model breaks, to prevent future outputs from being broken
    tmp_history = 0
    if step!=0:
        tmp_history = chat_history_ids.clone()
    else:
        tmp_history = torch.tensor([[]], dtype=int)
    
    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids#torch.cat([new_user_input_ids, tokenizer.encode("Your name is Chizuru" + tokenizer.eos_token, return_tensors='pt')], dim=-1) # original input was just new_user_input_ids. This mod doesn't quite work because stuff at the beginning of a conversation determines the flow of it. # Add history here, in the else clause, if doing that. Don't forget to encode.

    # generated a response while limiting the total chat history to 1000 tokens, 
    # other options for model.generate: 
    # n-gram penalty with no_repeat_ngram_size=2. # Prevents repeated text.
    # Beam search: num_beams=5 # Picks the word route that has the highest probability overall. num-beams allows more beams to be searched
    # random_sampling: do_sample=True # allows the model to occasionally pick tokens with low probabilities
    # temperature = 0.7 # determines how likely te model is to choose unlikely tokens. Higher temperature = more chaotic. Setting temperature to 0 gives greedy sampling, where the largest probability is picked each time.
    # top_k sampling reduces the things that can be selected from to the k most-probable options.
    # top_p = 0.92 | top p sampling takes the collection of tokens with the highest probabilities such that their sum is less than or equal to the specified probability.
    # BPE is byte-pair encoding. The compression mapping equivalence thing. I know what that is. If I forget, look it up.
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, 
    pad_token_id=tokenizer.eos_token_id, 
    no_repeat_ngram_size=3,
    num_beams=10,
    top_k = 4,
    penalty_alpha=0.6,
    # do_sample = True,
    # temperature=0.9,
    # do_sample=True,
    # top_p = 0.9,
    # temperature=0.6
    ) # Model.generate presumably returns the whole text, including the input. This is why the code works.
    step += 1
    # Detect if model is broken and clear last input if true.
    if (model_is_broken(chat_history_ids[:, bot_input_ids.shape[-1]:][0])):
        chat_history_ids = tmp_history
        # print("【Chizuru】: Huh?!") # This was very helpful in staying natural, but actually made debugging hard so commented out
        return("【ERROR】: MODEL IS CONFUSED (Happens often)")
    else:
        # Return the most recent bot response
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True) # no goddamn clue what this line does. I think it takes everything until the next eos_token but I'm not sure.
        emotion = get_most_likely_emotion(classifier(output)) # Gets an emotion to go with the output. When GUI implemented, Chizuru will change expression. Code that while the model is training.
        return(["【Chizuru】: {}".format(output),emotion])
        # t1 = time.time()
        # tts.tts_to_file(text=output,
        #                 speaker=tts.speakers[0],
        #                 language=tts.languages[0],
        #                 file_path="output.wav",
        #                 #speaker_wav="pathtowav.wav",
        #                 #language="jp"
        #                 )
        # playsound("output.wav")
        # print("Time elapsed to play sound: {}".format(time.time() - t1))


    # print(chat_history_ids) # Debug, view chat history tokens

# GUI #
width = 400
height = 510

window = Tk()
window.resizable(False,False)
# window.geometry(str(width)+"x"+str(height))
window.title("01Unit") # Reference to Alternative. 15 billion parameters in the palm of your hand! Except slightly fewer.
# frame = Frame(window,width=300,height=300)
# frame.pack()
# frame.place(anchor='center',relx=0.5,rely=0.5)
# img = ImageTk.PhotoImage(Image.open("test.png"))

def process_image(i): # i is string, path to image
    img = Image.open(i)
    img = img.resize((img.size[0],img.size[1]), resample=Image.ANTIALIAS) # copy the image so that we don't make the original way too small
    img.thumbnail((400,400))
    new_image = ImageTk.PhotoImage(img)
    return new_image


# img = Image.open("test.png")
# img = img.resize((img.size[0],img.size[1]), resample=Image.ANTIALIAS) # copy the image so that we don't make the original way too small
# # img_resized = img.resize((300,300))
# img.thumbnail((400,400))
default_img = process_image("default.png")#
happy = process_image("happy.png")#
angry = process_image("angry.png")#
sad = process_image("sad.png")
surprise = process_image("surprise.png")#
love = process_image("love.png")#
fear = process_image("fear.png")#

canvas = Canvas(window,width=400,height=450,bg='#000')
canvas.pack()

canvas.create_image(width/2,height/2,anchor=CENTER,image=default_img)
canvas.create_image(width/2,height/2,anchor=CENTER,image=happy)
def expression(emotion):
    canvas.delete("all")
    if emotion == "sadness":   
        canvas.create_image(width/2,height/2,anchor=CENTER,image=sad)
    elif emotion == "joy":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=happy)
    elif emotion == "love":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=love)
    elif emotion == "anger":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=angry)
    elif emotion == "fear":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=fear)
    elif emotion == "surprise":
        canvas.create_image(width/2,height/2,anchor=CENTER,image=surprise)
    else:
        print(emotion)
        canvas.create_image(width/2,height/2,anchor=CENTER,image=default_img)

# image_label = Label(frame,image = img)
# image_label.pack(fill = "x")
response = Label(window,text="",font=('Arial',17,'bold'),fg='#84F692',bg='#000')
response.pack(fill = "x")
chat_box = Entry(window,font=('Arial',17,'bold'),fg='#D5D5E8',bg='#000')
chat_box.pack(fill = "x")

def key_pressed(e):
    if (e.keysym == "Return"):
        generation = generate(chat_box.get())
        response.configure(text = generation[0])
        expression(generation[1])
        chat_box.configure(text="") # Clear written text

def entered_text():
    res = "REPLACE WITH MODEL CODE" + chat_box.get()
    response.configure(text = res) 

window.bind("<KeyRelease>",key_pressed)
# window.rowconfigure(11)
# window.columnconfigure(11)
# cols, rows = window.grid_size()

# for col in range(cols):
#     window.grid_columnconfigure(col,minsize=20)

# for row in range(rows):
#     window.grid_rowconfigure(row,minsize=20)

window.mainloop() 

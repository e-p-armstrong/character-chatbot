import torch
from transformers import pipeline
import numpy as np
from TTS.api import TTS
from playsound import playsound
import time
import os
import tkinter
os.environ["TOKENIZERS_PARALLELISM"] = "false"

classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

model_name = TTS.list_models()[0]
tts = TTS(model_name) # optional parameters here in case of custom model.

output_dir = 'output'
model_type = 'gpt2'
model_name_or_path = 'microsoft/DialoGPT-small'
config_name = 'microsoft/DialoGPT-small'
tokenizer_name = 'microsoft/DialoGPT-small'
cache_dir = 'cached'

# chizuru_bot = torch.from_pretrained("path/to/model")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelWithLMHead
import torch
import transformers

transformers.logging.set_verbosity_error() # Prevent it from bitching about padding_side

def get_most_likely_emotion(emotion_array):
    flatten = emotion_array[0]
    chances = []
    for e in flatten:
        chances.append(e['score'])
    highest_score = np.argmax(chances)
    return flatten[highest_score]['label'] # Returns most-likely emotion

def expression(emotion):
    if emotion == "sadness":
        return "(;_;)"
    elif emotion == "joy":
        return "\\(^-^)/"
    elif emotion == "love":
        return "(♥ω♥*)"
    elif emotion == "anger":
        return "(　ﾟДﾟ)＜!!"
    elif emotion == "fear":
        return "ヾ(ﾟдﾟ)ﾉ゛"
    elif emotion == "surprise":
        return "(⚆ᗝ⚆)"


# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side = 'left')
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium") # Will load from file once it's trained.
config = AutoConfig.from_pretrained(config_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained( # LM head, recall it from Andrej Karpathy's nanoGPT tutorial. I think it either refers to how transformers/masking/token probabilities are calculated, OR the linear models that help bend inputs/outputs into certain shapes.
        model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir,
    )

# device = torch.device("cuda")
# model.to(device)

def model_is_broken(output):
    if (output == ""): # This would be a machine learning model. Right now it's just a stub.
        return True
    return False

window = tkinter.Tk()
window.title("Chat!")



for step in range(10):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input("【 User 】: ") + tokenizer.eos_token, return_tensors='pt')


    # store history before input and output in a temp variable, incase model breaks
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
    # BPE is byte-pair encoding. I know what that is. If I forget, look it up.
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, 
    pad_token_id=tokenizer.eos_token_id, 
    no_repeat_ngram_size=3,
    num_beams=10,
    # do_sample = True,
    # temperature=0.2
    # do_sample=True,
    # top_p = 0.9,
    # temperature=0.6
    ) # Model.generate presumably returns the whole text, including the input. This is why the code works.
    
    # Detect if model is broken and clear last input if true.
    if (model_is_broken(chat_history_ids[:, bot_input_ids.shape[-1]:][0])):
        chat_history_ids = tmp_history
        print("【Chizuru】: Huh?!")
    else:
        # pretty print last ouput tokens from bot
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        emotion = get_most_likely_emotion(classifier(output)) # Gets an emotion to go with the output. When GUI implemented, Chizuru will change expression. Code that while the model is training.
        print("【Chizuru】: {}".format(output))
        print("Expression is: {}".format(expression(emotion)))
        t1 = time.time()
        tts.tts_to_file(text=output,
                        speaker=tts.speakers[0],
                        language=tts.languages[0],
                        file_path="output.wav",
                        #speaker_wav="pathtowav.wav",
                        #language="jp"
                        )
        playsound("output.wav")
        # print("Time elapsed to play sound: {}".format(time.time() - t1))


    # print(chat_history_ids) # Debug, view chat history tokens


# Idea: feed the AI information about itself by having a few statements "your name is X, your occupation is Y" before the user can interact
#       Append it to the chatlog. This way the AI will be able to answer questions that make sense about it.

    # Behold my analysis:
    # What's probably causing the repititon is that the bot is looking at the entire past conversation when generating every input, so if anything I write has a strong impact on
    # the generated output, it will continue to have a strong impact even after I type new stuff. This will make the bot choose the same output, in reply to my past input, making it repeat itself
    # Like a broken record.
    # solution: no_repeat_ngram_size
    # end result: I was right and the fix worked

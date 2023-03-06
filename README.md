# Using DialoGPT and the script of a visual novel to replicate a character as a chatbot

A chatbot, complete with a gui, that somewhat emulates the speech patterns of Sakaki Chizuru from the Muv Luv series of visual novels. 

What this is:
This is a chatbot that uses a fine-tuned version of Microsoft's DialoGPT model to produce outputs. It also uses a sentiment analysis model from Huggingface to detect the emotion of a response and set an expression on an image of the character represented in the GUI accordingly. I ripped most of the script from Muv Luv Extra's Chizuru route using Textractor (https://github.com/Artikash/Textractor) then trained the model to reply to input like it's Chizuru. Results are mixed at this stage: the chatbot can speak sensibly for a time, and also occasionally speaks like Chizuru, but lacks any knowledge of events that the character would know about and can also be VERY easily confused. These problems are not neccessarily unique to this fine-tuned version; DialoGPT, in its default state, can get easily confused too. I intend to transition this repo to using a different model (like GODEL) ASAP, to improve quality.

The trained model parameters themselves are too large to host here without LFS. You can train the model yourself by running process_and_model.py. You then chat with it by moving the "output" directory inside the "chat" folder (in scripts) and running "chat.py."


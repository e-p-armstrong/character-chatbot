# Adapted from https://towardsdatascience.com/make-your-own-rick-sanchez-bot-with-transformers-and-dialogpt-fine-tuning-f85e6d1f4e30
# and
# https://github.com/ncoop57/i-am-a-nerd/blob/master/_notebooks/2020-05-12-chatbot-part-1.ipynb 
def make_strings(strlst, tokenizer):
    flatten = lambda l: [item for sublist in l for item in sublist]
    context = []
    n = 7
    for i in range(n, len(strlst)):
        row = []
        prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses  
        for j in range(i, prev, -1):
            row.append(strlst[1][j])
        # Encode each string in row with tokenizer and concatenate together to make a single string with context and response
        conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        context.append(conv)
    return context
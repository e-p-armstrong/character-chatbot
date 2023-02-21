import os
import pandas as pd
import re



charname_regex = re.compile(r"【.+?】")
dialogue_regex = re.compile(r"「.+?」")
# re.search(str)

def makecols(str):
    charname_results = charname_regex.search(str)
    dialogue_results = dialogue_regex.search(str)
    if charname_results is None:
        return ['MONOLOGUE',dialogue_results]
    return [charname_results.group(0)[1:][:-1],dialogue_results.group(0)[1:][:-1]]


# str = "【Class Rep】「Not her! You, Shirogane-kun!! You're the one who always bursts in here at the last minute every day, kicking up a huge racket...*sigh*...」"
# print([charname_regex.search(str).group(0),dialogue_regex.search(str).group(0)])

script = list(map(makecols,open('muvluv-chizururoute-script.txt').read().replace('\n','').replace('\x05','').split('')))

# Match character name: (【.+?】)
# Match dialogue:       (「.+?」)

print(script) ## Next up: create pandas dataframe, take character name and put in a column, put text in a separate column. Then do context loop and put past 7 responses each in own column.

def construct_conv(row, tokenizer, eos = True):
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv
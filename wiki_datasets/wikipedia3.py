from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import math
import logging

"""
more similar to SimCSE, take a few consecutive sentences from every article
"""

# Shape (6458670, 4)
ds = load_dataset("wikipedia", "20220301.en")
f = open("wiki_datasets/wiki3/wiki3_3.txt","w")

logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

# counter to keep track of sentences written
c = 0

# iterate until we have 1M sentences
while True:

    # log to check if infinite loop
    if c % 1000 == 0:
        logging.info("check")

    # pick random article out of dataset
    # iterate until we have an article with more than 20 sentences (in short articles more than half the sentences might be not sentences but references etc.)
    # 20 instead of 10 like in other methods because here we take 4 sentences per article
    while True:
        index = random.randrange(6458670)
        sents = sent_tokenize(ds['train'][index]['text'],language='english')
        if (len(sents) >= 20):
            break
    
    # pick first 4 sentence in chosen article (not containing line breaks)
    # iterate until we have a sentence with word count of 10 to 50, only take sentences without line breaks
    # if can't find 4 sentences then break and check other articles
    found = 0
    tries = 0
    sindex = 0
    while found < 4:

        words = word_tokenize(sents[sindex])

        if len(words) > 10 and len(words) < 50 and "\n" not in sents[sindex]:
            f.write(sents[sindex])
            f.write("\n")
            found += 1
            c += 1
        sindex += 1

        if sindex == len(sents):
            break

    if c >= 1000000:
        break

f.close()
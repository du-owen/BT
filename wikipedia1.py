from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import math
import logging


""" 
method: remove "\n" from the text first and then sent_tokenize
"""

# Shape (6458670, 4)
ds = load_dataset("wikipedia", "20220301.en")
f = open("wiki_datasets/wiki1/wiki1_2.txt","w")

logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

# counter to keep track of sentences written
c = 0

# iterate until we have 1M sentences

while True:

    # log to check if infinite loop
    if c % 1000 == 0:
        logging.info("check")

    # pick random article out of dataset, remove all line breaks
    # iterate until we have an article with more than 10 sentences (in short articles more than half the sentences might be not sentences but references etc.)
    while True:
        index = random.randrange(6458670)
        sents = sent_tokenize(ds['train'][index]['text'].replace("\n",""),language='english')
        if (len(sents) >= 10):
            break

    # pick random sentence in chosen article, take only sentences in the first 60% because at the end of article there's mostly references
    # iterate until we have a sentence with word count of 10 to 50
    # if there is no such sentence we break out of loop and try another article
    found = False
    tries = 0

    while not found:
        
        sindex = random.randrange(math.floor(len(sents)*0.6))

        words = word_tokenize(sents[sindex])

        if len(words) > 10 and len(words) < 50:
            f.write(sents[sindex])
            f.write("\n")
            found = True
            c += 1

        tries += 1

        if tries > 50:
            break

    if c == 1000000:
        break

f.close()
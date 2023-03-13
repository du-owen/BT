from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import math
import logging

# Shape (6458670, 4)
ds = load_dataset("wikipedia", "20220301.en")
f = open("wiki_new3.txt","w")

logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

for i in range(250):
    if i % 1000 == 0:
        logging.info("check")
    while True:
        index = random.randrange(6458670)
        sents = sent_tokenize(ds['train'][index]['text'].replace("\n",""),language='english')
        if (len(sents) >= 10):
            break
    
    found = 0
    tries = 0
    sindex = 0
    while found < 4:

        words = word_tokenize(sents[sindex])

        if len(words) > 10 and len(words) < 50:
            f.write(sents[sindex])
            f.write("\n")
            found += 1
        sindex += 1

        if sindex == len(sents):
            break


f.close()
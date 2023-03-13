from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import math
import logging

# Shape (6458670, 4)
ds = load_dataset("wikipedia", "20220301.en")
f = open("wiki_new1.txt","w")

logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

# 1. remove "\n" from the text first and then sent_tokenize
# 2. sent_tokenize, and then only take sentences that don't contain \n
# 3. more similar to SimCSE, take a few consecutive sentences from every article
for i in range(1000):
    if i % 1000 == 0:
        logging.info("check")
    while True:
        index = random.randrange(6458670)
        sents = sent_tokenize(ds['train'][index]['text'].replace("\n",""),language='english')
        if (len(sents) >= 10):
            break
    
    found = False
    tries = 0
    while not found:
        
        sindex = random.randrange(math.floor(len(sents)*0.6))

        words = word_tokenize(sents[sindex])

        if len(words) > 10 and len(words) < 50:
            # if "\n" in sents[sindex]:
            #     # stemp = sents[sindex].split("\n")
            #     # if len(stemp) > 2:
            #     #     break
            #     # else:
            #     #     final = stemp[0] if len(stemp[0])>len(stemp[1]) else stemp[1]
            #     #     f.write(final)
            #     #     f.write("\n")
            #     #     found = True
            #     pass    
            # else:
            f.write(sents[sindex])
            f.write("\n")
            found = True

        tries += 1

        if tries > 50:
            break


f.close()
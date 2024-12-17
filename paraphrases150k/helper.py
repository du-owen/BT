import json
import openai
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import math
import time

job = 0

prompt = "Generate 5 new sentences, which are semantically similar but lexically and syntactically divergent from the following: "

openai.organization = ""
openai.api_key = ""

# file with indices where paraphrasing failed, need to create paraphrases of new sentences, insert them at the right place and update the original sentence file
with open("paraphrases150k/badIndices{}.json".format(job),"r") as ifile:
    indices = ifile.read()
    indices = json.loads(indices)

# file with paraphrases
with open("paraphrases150k/parap_trunc150k{}.json".format(job),"r") as pfile:
    pphrases = pfile.read()
    pphrases = json.loads(pphrases)

# wiki sets where we get new sentences from 
ds = load_dataset("wikipedia", "20220301.en")
newpphrases = []
c = 0

# file with original sentences
with open("wiki_datasets/wiki1/wiki1_1.txt","r") as sfile:
    origsents = sfile.readlines()

while True:

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
            cursent = sents[sindex]
            err = True
            while err:
                try:
                    completion = openai.ChatCompletion.create(
                        model = "gpt-3.5-turbo",
                        messages = [{"role": "user", "content": prompt + cursent}]
                    )
                    err = False
                except Exception as e:
                    err = True
                    time.sleep(10)
                    pass
            try:
                res = completion['choices'][0]['message']['content']
                res = res.split('\n')
                res = list(filter(lambda s:s!='',res))
                res = [x.split(' ', 1)[1] for x in res]
                if len(res) != 5:
                    continue
                newpphrases.append(res)
                found = True
                origsents[indices[c]] = cursent + "\n"
                c += 1
            except IndexError as e:
                continue

        tries += 1

        if tries > 50:
            break

    if c == len(indices):
        break

for pindex,lindex in enumerate(indices):
    pphrases.insert(lindex,newpphrases[pindex])

# overwrite old files
with open("wiki_datasets/wiki1/wiki1_1.txt","w") as sfile:
    sfile.writelines(origsents)

with open("paraphrases150k/parap_trunc150k{}.json".format(job),"w") as wpfile:
    pphrases = json.dumps(pphrases)
    wpfile.write(pphrases)

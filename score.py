from simcse import SimCSE
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import json
from datasets import load_dataset
import numpy as np

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

file = open("parap_trunc5.json","r")
j = file.read()
sents = json.loads(j)

wfile = open("parap_scores5.json","w")
wfile.write("{\"scores\":")
wfile.write("[")

dataset = load_dataset("text", data_files="wiki_trunc_new.txt", cache_dir="~/transformers_cache")

totalavg = 0
totalbleu = 0
totalsim = 0

for i in range(len(sents)):

    orig = dataset["train"][i]["text"]
    paraphrases = sents[i][:5]

    # Calculate similarity and normalize values
    sims = model.similarity([orig],paraphrases)[0]
    sims = list(map(lambda x:(x+1)/2,np.array(sims)))

    # Calculate bleu score
    refs = list(map(lambda x:word_tokenize(x),paraphrases))
    chencherry = SmoothingFunction()
    bleus = sentence_bleu(refs, word_tokenize(orig),smoothing_function=chencherry.method1)


    # Calculate metric
    meansim = np.mean(sims)
    eval_score = (meansim + (1-bleus))/2
    totalbleu += bleus
    totalsim += meansim

    # Write results into JSON
    scores = {"meansim": meansim, "meanbleu": bleus, "combined": eval_score, "simscores": sims}
    scores = json.dumps(scores)
    wfile.write(scores)

    # Update scores across whole dataset
    totalavg += eval_score

    # If last sentence then we don't need a comma in JSON format anymore
    if i == len(sents)-1:
        continue
    
    wfile.write(",\n")

# Finish writing dataset scores into JSON, access with s["total"]["max"|"min"|"mean"] 
wfile.write("],\n\"total\":")
totalavg /= len(sents)
totalbleu /= len(sents)
totalsim /= len(sents)
totals = {"mean":totalavg, "sim": totalsim, "bleu": totalbleu}
totals = json.dumps(totals)
wfile.write(totals)
wfile.write("}")
wfile.close
        
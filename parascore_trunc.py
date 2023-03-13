from parascore import ParaScorer
import json
from datasets import load_dataset
import numpy as np

# ParaScorer object
scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased')

# Read paraphrased sentences from json file
file = open("parap_trunc.json","r")
j = file.read()
sents = json.loads(j)

# Create json file for parascores, access scores of sentence i with s["scores"][i]["min"|"max"|"mean"|"all"|"std"]
wfile = open("parascores1.json","w")
wfile.write("{\"scores\":")
wfile.write("[")

# Load original sentences
dataset = load_dataset("text", data_files="wiki_trunc.txt", cache_dir="~/transformers_cache")

# Keep track of averages, minimum and maximum scores across whole dataset
totalavg = 0
totalmin = 1
totalmax = 0

# We compare every sentence with each of its 20 paraphrases and evaluate the Parascore
for i in range(1000):

    allscores = []

    # If the original input is not paraphrasable (e.g. a singular noun) then we skip
    if len(sents[i]) < 20:
        wfile.write("{},\n")
        continue

    # Evaluate every paraphrase of a given sentence
    for j in range(20):
        score = scorer.free_score([sents[i][j]],[dataset["train"][i]['text']])[0]
        allscores.append(score.item())

    # Write results into JSON
    maxscore = np.max(allscores)
    minscore = np.min(allscores)
    meanscore = np.mean(allscores)
    scores = {"max":maxscore, "min":minscore, "mean": meanscore, "all":allscores, "std": np.std(allscores)}
    scores = json.dumps(scores)
    wfile.write(scores)

    # Update scores across whole dataset
    totalavg += meanscore
    totalmin = totalmin if totalmin < minscore else minscore
    totalmax = totalmax if totalmax > maxscore else maxscore

    # If last sentence then we don't need a comma in JSON format anymore
    if i == 999:
        continue
    
    wfile.write(",\n")

# Finish writing dataset scores into JSON, access with s["total"]["max"|"min"|"mean"]
wfile.write("],\n\"total\":")
totalavg /= 1000
totals = {"max":totalmax, "min":totalmin, "mean":totalavg}
totals = json.dumps(totals)
wfile.write(totals)
wfile.write("}")
wfile.close
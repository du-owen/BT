import json
import matplotlib.pyplot as plt
from datasets import load_dataset

file = open("parascores.json","r")
j = file.read()
scores = json.loads(j)

data = [[],[]]

dataset = load_dataset("text", data_files="wiki_trunc.txt", cache_dir="~/transformers_cache")

# Correlation between length of sentence and score
data[0] = [score.get("mean") for score in scores.get("scores",[]) if score.get("mean") is not None]
data[1] = [len(dataset["train"][i]['text']) for i in range(len(dataset["train"])) if len(dataset["train"][i]['text']) > 1]

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

# Scatter plot
plt.scatter(data[0], data[1])

plt.savefig('corr1.png')


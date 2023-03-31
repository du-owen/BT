import json

file = open("paraphrases150k/parap_trunc150k0.json","r")
sents = file.read()
sents = json.loads(sents)

print(len(sents))
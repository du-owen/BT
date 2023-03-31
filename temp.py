import json

with open("paraphrases150k/badIndices1.json","r") as f:
    sents = f.read()
    sents = json.loads(sents)

print(len(sents))


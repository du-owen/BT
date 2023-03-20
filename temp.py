from simcse import SimCSE
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
sentences_b = ['He plays guitar.', 'A woman is making a photo.']
similarities = model.similarity(sentences_a, sentences_b)

similarities = np.array(similarities)

print(list(map(lambda x: x, similarities)))

# embeddings = model.encode("A woman is reading.")

# print(embeddings)

# cand = 'this is a dog'.split()


# ref = ["that's a dog".split()]

# print('BLEU score -> {}'.format(sentence_bleu(ref, cand)))

# reference = [
#     'this is a dog'.split(),
#     'it is dog'.split(),
#     'dog it is'.split(),
#     'a dog, it is'.split() 
# ]
# candidate = 'it is dog'.split()
# print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights = [0.33,0.33,0.33,0])))

# candidate = 'it is a dog'.split()
# print('BLEU score -> {}'.format(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))))
from datasets import load_dataset
import openai
import json
import logging
import time

# OpenAI initialization
openai.organization = "org-ttEh7YxIVc0VaC1LXt8xVuIO"
openai.api_key = "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"

# Load sentences
dataset = load_dataset("text", data_files="wiki_trunc_new.txt", cache_dir="~/transformers_cache")

# Choose paraphrasing prompt for ChatGPT
# prompt 1: Generate 5 paraphrases of the following:
# prompt 2: Generate 5 new sentences, which are semantically similar but lexically and syntactically divergent from the following:
# prompt 3: I want you to act as a paraphrasing tool. I will provide you a sentence and your task is to generate 5 paraphrases. These will act as augmented data that I will use to train a sentence embedding model evaluated on a semantic text similarity task. The sentence is:
# prompt 4: On a scale of 1 to 5, where 1 is the most semantically similar but least lexically divergent and 5 is the least semantically similar but most lexically divergent, generate a paraphrase for each scale of the following: 
# prompt 5: Generate 5 paraphrases, where the first paraphrase has the highest semantic similarity but the lowest lexical divergence and the last paraphrase has the lowest semantic similarity and the hihgest lexical divergence, of the following:

prompt = "Generate 5 paraphrases, where the first paraphrase has the highest semantic similarity but the lowest lexical divergence and the last paraphrase has the lowest semantic similarity and the hihgest lexical divergence, of the following: "

# Create json file for paraphrased sentences, access j-th paraphrase for i-th sentence with s[i][j]
file = open("parap_trunc5.json","w")

# Keep track of total tokens used
totaltk = 0

# Log API requests and answers
logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

file.write("[")

for i in range(10):#len(dataset["train"])):
    logging.info(i)

    # API request, if we get API error then we try again after sleeping 10s (not checked if it 100% works)
    err = True
    while err:
        try:
            completion = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [{"role": "user", "content": prompt + dataset["train"][i]['text']}]
            )
            err = False
        except Exception as e:
            logging.warning(e)
            err = True
            time.sleep(10)
            pass
    
    # Track tokens
    totaltk += completion['usage']["total_tokens"]

    # Parse the result into separate paraphrases
    res = completion['choices'][0]['message']['content']
    res = res.split('\n')
    res = list(filter(lambda s:s!='',res))
    res = [x.split(' ', 1)[1] for x in res]

    # Write to JSON file
    jsonr = json.dumps(res)
    file.write(jsonr)

    # in last iteration no comma
    if i == len(dataset["train"])-1:
        continue

    file.write(",\n")
    

print(totaltk/len(dataset["train"]))
file.write("]")
file.close()
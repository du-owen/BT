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
prompt = "Generate 5 new sentences, which are semantically similar but lexically and syntactically divergent from the following: "

# Create json file for paraphrased sentences, access j-th paraphrase for i-th sentence with s[i][j]
file = open("parap_trunc2.json","w")

# Keep track of total tokens used
totaltk = 0

# Log API requests and answers
logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

file.write("[")

for i in range(len(dataset["train"])):
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
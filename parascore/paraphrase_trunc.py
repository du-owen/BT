from datasets import load_dataset
import openai
import json
import logging
import time

# OpenAI initialization
openai.organization = "org-ttEh7YxIVc0VaC1LXt8xVuIO"
openai.api_key = "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"

# Load sentences
dataset = load_dataset("text", data_files="wiki_trunc.txt", cache_dir="~/transformers_cache")

# Choose paraphrasing prompt for ChatGPT
prompt = "Generate 20 paraphrases of the following: "

# Create json file for paraphrased sentences, access j-th paraphrase for i-th sentence with s[i][j]
file = open("parap_trunc2.json","w")

# Keep track of total tokens used
totaltk = 0

# Log API requests and answers
logging.basicConfig(filename='out/cur2.log', level = logging.DEBUG)

file.write("[")

for i in range(1000):
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
        except openai.error.APIError as e:
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
    if i == 999:
        continue
    file.write(",\n")
    

print(totaltk/1000)
file.write("]")
file.close()
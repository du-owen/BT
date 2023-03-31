from datasets import load_dataset
import openai
import json
import logging
import time
from multiprocessing import Pool
import os


job = int(os.environ.get("SLURM_ARRAY_TASK_ID")) # each job processes 15k paraphrases

# returns list of paraphrases and list of indices where paraphrase wasn't possible (handled later in helper.py)
def pphrases(interval):
    start, end = interval
    results = []
    badIndices = []
    for i in range(start,end):
        logging.info(i)
        # API request, if we get API error then we try again after sleeping 10s (not checked if it 100% works)
        err = True
        while err:
            try:
                completion = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = [{"role": "user", "content": prompt + dataset[i]}]
                )
                err = False
            except Exception as e:
                logging.warning(e)
                err = True
                time.sleep(10)
                pass

        # Parse the result into separate paraphrases
        try:
            res = completion['choices'][0]['message']['content']
            res = res.split('\n')
            res = list(filter(lambda s:s!='',res))
            res = [x.split(' ', 1)[1] for x in res]

            if len(res) != 5:
                badIndices.append(i)
                continue

            results.append(res)
        except IndexError as e:
            badIndices.append(i)
            pass

    return results, badIndices

def main():
    # OpenAI initialization
    openai.organization = "org-ttEh7YxIVc0VaC1LXt8xVuIO"
    openai.api_key = "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"


    # Create json file for paraphrased sentences, access j-th paraphrase for i-th sentence with s[i][j]
    file = open("paraphrases150k/parap_trunc150k{}.json".format(job),"w")

    # Load sentences
    global dataset
    global prompt
    dataset = load_dataset("text", data_files="wiki_datasets/wiki1/wiki1_1.txt", cache_dir="~/transformers_cache")["train"][:150000]["text"]
    prompt = "Generate 5 new sentences, which are semantically similar but lexically and syntactically divergent from the following: "
    
    numProcesses = 15
    intervalsize=1000 # 1000 paraphrases per process
    intervals = [(i,i+intervalsize) for i in range(job*15000+0,job*15000+numProcesses*intervalsize,intervalsize)] # interval range depends on job number

    # every process works on a separate interval
    with Pool(processes=numProcesses,initializer=loginit) as pool:
        results = pool.map(pphrases, intervals)
        pool.close()
        pool.join()

    # parse result, paraphrases contains paraphrases, indices contains lines where paraphrase wasn't possible, handled in helper.py
    paraphrases = [x[0] for x in results]
    paraphrases = [item for sublist in paraphrases for item in sublist]

    indices = [x[1] for x in results]
    indices = [item for sublist in indices for item in sublist]

    jsonr = json.dumps(paraphrases)
    file.write(jsonr)
    file.close()

    with open("paraphrases150k/badIndices{}.json".format(job),"w") as ifile:
        jsonr = json.dumps(indices)
        ifile.write(jsonr)

def loginit():
    # Log API requests
    logging.basicConfig(filename='out/pp150k{}.log'.format(job), level = logging.DEBUG)


if __name__ == "__main__":
    main()
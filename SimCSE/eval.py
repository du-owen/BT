from prettytable import PrettyTable

t = PrettyTable(["dataset","dropout","epochs","mode","lr","batchsize","STS12","STS13","STS14","STS15","STS16","STSBenchmark","SICKRelatedness","Avg"])
t.add_row(["simcse150k",0.1,5,"unsup simcse",3e-5,64,66.57, 81.41, 72.74, 80.32, 77.81, 74.5, 70.19, 74.79])
t.add_row(["simcse1m",0.1,1,"unsup simcse",3e-5,64,68.17, 81.37, 73.35, 80.0, 77.58, 76.64, 69.9, 75.29])
t.add_row(["simcse1m",0.1,5,"unsup simcse",3e-5,64,66.14, 80.71, 72.49, 79.89, 78.58, 76.77, 71.74, 75.19])
t.add_row(["wiki1m",0.1,1,"unsup simcse",3e-5,64,68.78, 81.59, 75.19, 81.44, 77.41, 78.5, 72.22, 76.45])
t.add_row(["wiki1m",0.1,5,"unsup simcse",3e-5,64,65.18, 78.62, 70.14, 79.74, 77.61, 75.66, 70.96, 73.99])
t.add_row(["wiki150k",0.1,1,"unsup simcse",3e-5,64,60.26, 74.21, 64.45, 74.68, 74.96, 68.88, 67.52, 69.28])
t.add_row(["wiki150k",0.1,5,"unsup simcse",3e-5,64,67.25, 76.85, 71.98, 80.26, 77.35, 77.39, 70.61, 74.53])
t.add_row(["wiki150k",0.1,5,"paraphrase + mlp for predicting",3e-5,64,67.71, 76.89, 70.02, 77.02, 77.97, 74.44, 72.31, 73.77])
t.add_row(["wiki150k",0.1,2,"only 2 paraphrases",3e-5,64,68.7, 78.67, 70.7, 78.19, 78.85, 75.39, 71.39, 74.56])
t.add_row(["wiki150k",0.1,5,"paraphrase",3e-5,64,70.05, 77.31, 70.42, 76.93, 77.22, 75.22, 71.09, 74.03])
t.add_row(["wiki150k",0.1,10,"paraphrase",3e-5,64,68.71, 79.71, 71.12, 77.71, 79.27, 75.97, 72.03, 74.93])
t.add_row(["wiki150k",0.1,15,"paraphrase",3e-5,64,68.07, 79.13, 70.59, 77.09, 78.89, 75.22, 72.13, 74.45])
t.add_row(["wiki150k",0.1,20,"paraphrase",3e-5,64,69.64, 79.26, 71.3, 77.79, 79.15, 76.15, 72.31, 75.09])
t.add_row(["wiki150k",0.1,25,"paraphrase",3e-5,64,68.41, 79.24, 70.36, 77.89, 79.21, 75.24, 72.14, 74.64])
t.add_row(["wiki150k",0.1,5,"paraphrase",5e-5,256,69.57, 78.1, 70.08, 77.5, 78.54, 75.07, 71.82, 74.38])

t.add_row(["wiki150k",0.0,5,"paraphrase",5e-5,256,68.34, 78.42, 70.07, 77.65, 78.22, 74.45, 71.54, 74.1])
t.add_row(["wiki150k",0.0,2,"only 2 paraphrases",3e-5,64,68.55, 80.1, 71.8, 78.39, 78.88, 76.22, 71.81, 75.11])
t.add_row(["wiki150k",0.0,5,"paraphrase",3e-5,64,69.23, 78.38, 71.32, 78.84, 78.44, 75.59, 71.99, 74.83])
t.add_row(["wiki150k",0.0,10,"paraphrase",3e-5,64,69.5, 80.41, 72.18, 78.76, 78.88, 76.41, 71.77, 75.42])
t.add_row(["wiki150k",0.0,15,"paraphrase",3e-5,64,68.84, 80.2, 72.18, 77.75, 78.86, 76.27, 71.86, 75.14])
t.add_row(["wiki150k",0.0,20,"paraphrase",3e-5,64,68.48, 80.12, 71.96, 77.63, 78.98, 75.79, 71.74, 74.96])

tr = PrettyTable(["dataset","dropout","epochs","mode","lr","batchsize","STS12","STS13","STS14","STS15","STS16","STSBenchmark","SICKRelatedness","Avg"])
tr.add_row(["simcse benchmark","-","-","-","-","-",70.16,81.77,73.24,81.36,80.65,80.22,68.56,76.57])
tr.add_row(["wiki150k",0.1,5,"paraphrase",3e-5,64,71.89, 81.44, 72.43, 80.01, 80.32, 79.66, 74.11, 77.12])
tr.add_row(["wiki150k",0.1,10,"paraphrase",3e-5,64,70.84, 80.76, 71.81, 78.22, 80.46, 79.81, 74.58, 76.64])
tr.add_row(["wiki150k",0.1,15,"paraphrase",3e-5,64,70.82, 81.45, 71.76, 79.36, 79.45, 79.75, 74.57, 76.74])
tr.add_row(["wiki150k",0.1,20,"paraphrase",3e-5,64,68.94, 78.18, 71.29, 78.75, 79.74, 78.68, 73.4, 75.57])

tr.add_row(["wiki150k",0.0,5,"paraphrase",3e-5,64,69.41, 81.43, 71.86, 79.58, 79.66, 78.02, 75.35, 76.47])
tr.add_row(["wiki150k",0.0,10,"paraphrase",3e-5,64,69.84, 81.43, 72.54, 78.94, 79.88, 78.21, 75.73, 76.65])
tr.add_row(["wiki150k",0.0,15,"paraphrase",3e-5,64,70.37, 81.16, 72.47, 80.7, 79.96, 79.04, 76.03, 77.1])
tr.add_row(["wiki150k",0.0,20,"paraphrase",3e-5,64,68.94, 78.18, 71.29, 78.75, 79.74, 78.68, 73.4, 75.57])

tvic = PrettyTable(["dataset","dropout","epochs","mode","lr","batchsize","STS12","STS13","STS14","STS15","STS16","STSBenchmark","SICKRelatedness","Avg","ChatGPT Avg"])

tvic.add_row(["vicuna",0.0,10,"bert",3e-5,64,71.12, 78.57, 72.78, 79.02, 78.97, 76.92, 70.0, 75.34,75.42])
tvic.add_row(["vicuna",0.1,20,"bert",3e-5,64,70.11, 78.7, 71.59, 77.31, 78.51, 76.36, 71.42, 74.86,75.09])
tvic.add_row(["vicuna",0.1,5,"roberta",3e-5,64,69.46, 78.98, 71.09, 78.74, 79.01, 78.8, 72.53, 75.52,77.12])
tvic.add_row(["vicuna",0.1,25,"roberta",3e-5,64,71.86, 78.78, 71.44, 78.5, 78.27, 77.81, 72.3, 75.57,"-"])
tvic.add_row(["vicuna",0.0,15,"roberta",3e-5,64,69.39, 79.83, 71.48, 78.94, 78.35, 77.61, 72.97, 75.51,77.1])
tvic.add_row(["vicuna",0.0,20,"roberta",3e-5,64,70.41, 80.18, 71.63, 79.63, 78.6, 77.94, 73.18, 75.94,"-"])


with open('evals_table.txt', 'w') as f:
    print("dataset: wiki is own dataset, simcse is simcse dataset", file=f)
    print("mode: unsup simcse or paraphrase", file=f)
    print(t, file=f)
    print("following with roberta-base:",file=f)
    print(tr,file=f)
    print("following with vicuna generated paraphrases",file=f)
    print(tvic,file=f)
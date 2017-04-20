import sys
import numpy as np

def read_nbest(nbest_file, normalize=False):
    result = []
    scores = []
    cnt = 0
    for line in open(nbest_file):
        if not line.strip():
            cnt += 1
            continue
        pred, score = line.split("\t")
        if not pred.strip(): continue
        score = float(score)
        if len(result) == cnt:
            result.append([])
            scores.append([])
        result[cnt].append((pred,score))
        scores[cnt].append(score)
    if normalize:
        for i in range(len(result)):
            scores_arr = np.exp(scores[i])
            mean = np.mean(scores_arr)
            std = np.std(scores_arr)
            for j in range(len(result[i])):
                result[i][j] = (result[i][j][0],scores_arr[j]/(len(result[i])*mean*std))
    return result

normalize = "-normalize" in sys.argv
num_lang = 0
nbest_list = []
for nbest_file in sys.argv[1:]:
    if nbest_file != "-normalize":
        nbest_list.append(read_nbest(nbest_file, normalize))
        num_lang += 1
num_inst = len(nbest_list[0])

for i in range(num_inst):  
    m = {}
    for j in range(num_lang):
        for pred, score in nbest_list[j][i]:
            if pred not in m:
                m[pred] = []
            m[pred].append(score)
    merged_nbest = sorted(m.keys(), key=lambda x: max(m[x]), reverse=True)
    result = merged_nbest[0]
    print result

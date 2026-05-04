import csv
import numpy as np
import json

filename = "results/stoplight_BR-NS_1000.csv"

with open(filename, "r") as f:
    reader = csv.reader(f)
    avgs1, avgs2 = [], []
    for row in reader:
        if len(row) == 0:
            continue
    #     row = json.load(row)
        time_steps, avg1, avg2, last1, last2, actions = row
        avgs1.append(float(avg1))
        avgs2.append(float(avg2))
    # print(avgs1)
    print("Means:", np.mean(avgs1), np.mean(avgs2))
    print("Variance:", np.var(avgs1), np.var(avgs2))
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fitter import Fitter
from scipy.stats import exponnorm
np.random.seed(0)

pairs = pd.read_csv("../silver/external_pairs.csv", header=0, sep="\t")

# get pairs related to recall calculation
recall_pairs = pairs[pairs["split"].isnull()]
recall_pairs = recall_pairs[["h_name", "t_name"]]
n_sample = len(recall_pairs)
print(n_sample)


def resample(predictions):
    # resample
    f = Fitter(predictions["probability"], distributions=['exponnorm'])  # 创建Fitter类
    f.fit()
    f.summary()
    plt.show()

    exponnorm_dist = exponnorm(*f.fitted_param['exponnorm'])
    resampled_data = exponnorm_dist.rvs(size=n_sample)

    plt.hist(resampled_data, bins=30, density=True, alpha=0.6, color='g', label='Resampled Data Histogram')
    plt.hist(predictions["probability"], bins=30, density=True, alpha=0.6, color='b', label='Original Data Histogram')

    # plt.title('Enhancer Resampled Data Histogram')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return resampled_data


crispr = pd.read_csv("../results/predictions_CRISRP.csv", header=0, sep="\t")
lncrna_pairs = pd.read_csv("../results/predictions_enhancer.csv", header=0, sep="\t")
predictions = pd.concat([crispr, lncrna_pairs], axis=0)

train_threshold = predictions[predictions["source"] == "train"]["probability"].mean()


# keep prediction with largest probability
# idx = predictions.groupby(['query_node', 'prediction_node'])['probability'].idxmax()
# predictions = predictions.loc[idx]

# shuffle the t to get random pairs
random.seed(7)
h_name = random.choices(list(set(recall_pairs["h_name"])), k=len(recall_pairs))
t_name = random.choices(list(set(recall_pairs["t_name"])), k=len(recall_pairs))

# h_name = recall_pairs["h_name"].sample(frac=1).reset_index(drop=True)
# t_name = recall_pairs["t_name"].sample(frac=1).reset_index(drop=True)
random_pairs = pd.DataFrame({"h_name": h_name, "t_name": t_name})


def binom_recall(p):
    n_samples = n_sample
    samples = np.random.binomial(1, 1-p, n_samples)
    return np.sum(samples) / n_samples


def calculate_recall(threshold, recall_pairs, predictions):
    # join to get probabilities of recall pairs
    recall_pairs = pd.merge(recall_pairs, predictions, how="left", left_on=["h_name", "t_name"],
                            right_on=["query_node", "prediction_node"])
    recall_pairs = recall_pairs[predictions.columns]

    # ...
    thresholded_recall_pairs = recall_pairs.groupby(["query_node", "prediction_node", "prediction_node_type"])[
        "probability"].max()

    # ...
    result_df = pd.DataFrame({"max_probability": thresholded_recall_pairs})

    result_df["positive"] = result_df["max_probability"] > threshold

    # calculate the recall
    tp = len(result_df[result_df["positive"]])
    n = len(thresholded_recall_pairs)
    recall = tp / n

    return recall


def resample_recall(threshold, resampled_data):
    binary = np.where(resampled_data > threshold, 1, 0)
    return np.sum(binary) / len(binary)


resampled_data = resample(predictions)


x = np.linspace(0, 1, 100)
y = [calculate_recall(i, recall_pairs, predictions) for i in x]
plt.plot(x, y, label="NBFNet")
y_resample = [resample_recall(threshold=i, resampled_data=resampled_data) for i in x]
plt.plot(x, y_resample, label="exponnorm resample")

plt.axvline(x=train_threshold, color='green', alpha=0.6, linestyle='--')
plt.text(train_threshold+0.05, 0.45, f'p = {round(train_threshold, 2)}', fontsize=12)
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.legend()
plt.show()

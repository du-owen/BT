import numpy as np
import json
import matplotlib.pyplot as plt

# function that returns a boolean mask m, where m[i] is true if points[i] is NOT an outlier
# median absolute deviation
def mad_based_outlier(points, thresh):

    if len(points.shape) == 1:
        points = points[:,None]

    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score <= thresh

# Read parascores input
file = open("parascores.json","r")
j = file.read()
scores = json.loads(j)

# Get mean scores in a list
means = [score.get("mean") for score in scores.get("scores",[]) if score.get("mean") is not None]

# Calculate mean scores after pruning outliers
mask = mad_based_outlier(np.array(means),2)
means_pruned = np.array(means)[mask]
print(np.mean(means_pruned),len(means_pruned),len(means))

# Plot both distributions
fig,axes = plt.subplots(nrows=2)
bins = [(i/100) for i in range(101)]
axes[0].hist(means, bins=bins, edgecolor='black')
axes[0].set_title('Distribution of Parascores')
axes[0].set_xlabel('Mean Scores')
axes[0].set_ylabel('Frequency')

axes[1].hist(means_pruned, bins=bins, edgecolor='black')
axes[1].set_title('Distribution of Parascores after Removing Outliers')
axes[1].set_xlabel('Mean Scores')
axes[1].set_ylabel('Frequency')

fig.subplots_adjust(hspace=0.5)

plt.savefig('histo1.png')
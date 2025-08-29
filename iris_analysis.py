import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load iris dataset
from sklearn.datasets import load_iris
#!%matplotlib ipympl

# %%
iris = load_iris()

# map target names to target and clasify the different iris
names = iris.target_names[iris.target]
features = iris.feature_names


# Prepare dataframe
df_iris = pd.DataFrame(data = iris.data, columns=features)
df_iris["species"] = names

# %%
print("Check general dataset information")
print("-"*35)
print(df_iris.describe())


# %%
print("="*35)
print("\n\nCheck species distribution")
print("-"*35)
print(df_iris.groupby("species").size())


# %%
# Plot the different attributes of the dataset to check distributions
fig1 = plt.figure(figsize=(7,5))

sns.violinplot(
    data = df_iris,
    orient="h"
)

plt.tight_layout()
plt.show()

##############################################################
# %%
# Plot histograms for the different attributes
df_iris.hist(figsize=(6,6))

plt.tight_layout()
plt.show()

print("It looks like perhaps two of the input variables have a Gaussian\n" \
    "distribution. This is useful to note as we can use algorithms\n" \
    "that can exploit this assumption.")


# %%
# Same idea but plot using Seaborn and separate the species in each subplot
# First translate the dataframe to long format
df_long = df_iris.melt(id_vars="species", value_vars=features, var_name="feature", value_name="size")

# Check the distributions of each different species
g = sns.FacetGrid(df_long, col="feature", col_wrap=2, sharex=False, sharey=False, hue="species")
g.map(sns.histplot, "size", kde=True)

g.add_legend()
g.tight_layout()

print("It looks like the features of all species display a Gausian distribution.")

plt.show()

############################################################
# %%
# Multivariate plots
print("Check interactions between the variables. Plot attribute pairs.")
pp = sns.pairplot(df_iris, hue="species", diag_kind="kde", height=2)
# Could have used pd.plotting.scatter_matrix(df_iris)

pp.tight_layout()
plt.show()



# %%






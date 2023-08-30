import pandas as pd
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns


with open("po1_data.txt", "r") as f:
  f = f.readlines()


nested_list = [line.split(',') for line in f]
nested_list = [[float(item) if '.' in item else int(item) for item in row] for row in nested_list]
for row in nested_list[:5]:
    print(row)


columns = [
    "Subject Identifier",
    "Jitter (1)", "Jitter (2)", "Jitter (3)", "Jitter (4)", "Jitter (5)",
    "Shimmer (1)","Shimmer (2)", "Shimmer (3)", "Shimmer (4)", "Shimmer (5)", "Shimmer (6)", 
    "Harmonicity (1)", "Harmonicity (2)",  "Harmonicity (3)", 
    "Pitch (1)", "Pitch (2)", "Pitch (3)", "Pitch (4)", "Pitch (5)", 
    "Pulse (1)","Pulse (2)", "Pulse (3)", "Pulse (4)",
     "Voice (1)","Voice (2)", "Voice (3)", "UPDRS",
    "PD Indicator"
]

df = pd.DataFrame(nested_list, columns=columns)

print(df.head())



#Block - 4
data = df

# Drop irrelevant columns
data.drop(['Subject Identifier'], axis=1, inplace=True)

# Separate features and target
X = data.drop('PD Indicator', axis=1)  
y = data['PD Indicator']

# Descriptive Analysis
grouped_stats = X.groupby(y).mean()
print(grouped_stats)

# Visualize feature distributions
for col in X.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data, x=col, hue=y, bins=20, kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=10)  # Use f_classif for classification tasks
X_new = selector.fit_transform(X, y)

# Get the selected features' indices
selected_indices = selector.get_support(indices=True)

# Print the names of selected features
selected_features = X.columns[selected_indices]
print("Selected Features:", selected_features)

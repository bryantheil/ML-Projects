#%%
import pandas as pd

# Read in dataset
file = '/Users/bryantheil/Downloads/customers_dataset.csv'
df = pd.read_csv(file)
df.head()
#%%
# check for empty rows
sumofnull = df.isnull().sum()
sumofnull
#%%
# examine data types
print("Dataset datatypes:", df.dtypes)
print("*"*80)
print("Data description", df.describe())
print("*"*80)
print("Data shape", df.shape)
#%%
# analyzing target distribution
distribute = df['y'].value_counts()
distribute
#%%
# check duplicates
duplicates = df[df.duplicated()]

# drop duplicate data
df = df.drop_duplicates()

# check shape
print("Shape of dataset after dropping duplicates:", df.shape)
#%%
# visualize target variable distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="y", data=df, palette="pastel", hue="y", legend=False)
plt.title("Count Plot of 'y'")
plt.xlabel("y")
plt.ylabel("Count")
plt.show()
#%%
# visualize the job distribution
sns.countplot(x="job", data=df, palette="viridis", hue="job", legend=False)
plt.title("Count Plot of 'job'")
plt.xlabel("job")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()
#%%
# visualize marital distribution
sns.countplot(x="marital", data=df, palette="pastel", hue="marital", legend=False)
plt.title("Count Plot of 'marital'")
plt.xlabel("marital status")
plt.ylabel("Count")
plt.show()
#%%
# visualize education distribution
sns.countplot(x="education_qual", data=df, palette="viridis", hue="education_qual", legend=False)
plt.title("Count Plot of 'education'")
plt.xlabel("education level")
plt.ylabel("Count")
plt.show()
#%%
# visualize call type distribution
sns.countplot(x="call_type", data=df, palette="pastel", hue="call_type", legend=False)
plt.title("Count Plot of 'call_type'")
plt.xlabel("default")
plt.ylabel("Count")
plt.show()
#%%
# visualize month distribution
sns.countplot(x="mon", data=df, palette="viridis", hue="mon", legend=False)
plt.title("Count Plot of 'month'")
plt.xlabel("month")
plt.ylabel("Count")
plt.show()
#%%
# visualize previous outcome distribution
sns.countplot(x="prev_outcome", data=df, palette="pastel", hue="prev_outcome", legend=False)
plt.title("Count Plot of 'previous outcome'")
plt.xlabel("previous outcome")
plt.ylabel("Count")
plt.show()
#%%
# encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
categorical_columns = ['job', 'marital', 'education_qual', 'call_type', 'mon', 'prev_outcome','y']
# loop through categorical_columns and encode each column
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
df
#%%
# address class imbalance in target using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X = df.drop('y', axis=1)
y = df['y']
X_res, y_res = smote.fit_resample(X, y)

# shape of X_res
print("Shape of X_res:", X_res.shape)
# shape of y_res
print("Shape of y_res:", y_res.shape)
# value count of y_res
print("Value count of y_res:", y_res.value_counts())

#%%
# standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
# shape of X_scaled
print("Shape of X_scaled:", X_scaled.shape)
#%%
# splitting the dataset for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)
# shape of y_train
print("Shape of y_train:", y_train.shape)
#%%
# train and evaluate using the naive bayes model
# import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
# define the model
gnb = GaussianNB()
# fit the model
gnb.fit(X_train, y_train)
# predict
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy *100)
print("Precision:", precision *100)
print("Recall:", recall *100)
print("F1 Score:", f1 * 100)

#%%
# train and evaluate using the Linear Discriminant Analysis (LDA) Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy *100)
print("Precision:", precision *100)
print("Recall:", recall *100)
print("F1 Score:", f1 * 100)
#%%
# train and evaluate with the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy *100)
print("Precision:", precision *100)
print("Recall:", recall *100)
print("F1 Score:", f1 * 100)
#%%
# train and evaluate with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)
y_pred = etc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy *100)
print("Precision:", precision *100)
print("Recall:", recall *100)
print("F1 Score:", f1 * 100)
#%%
# train and evaluate with a RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy *100)
print("Precision:", precision *100)
print("Recall:", recall *100)
print("F1 Score:", f1 * 100)
#%%
# visualize a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# visualize confusion matrix using heatmap
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#%%

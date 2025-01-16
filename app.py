import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Upload Data Section
st.title('Web-based Data Mining and Analysis Application')

st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV, Excel, or TSV file", type=["csv", "xlsx", "tsv"])

if uploaded_file.name.endswith('.csv'):
	df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith('.xlsx'):
	df = pd.read_excel(uploaded_file)
elif uploaded_file.name.endswith('.tsv'):
	df = pd.read_csv(uploaded_file, sep='\t')

st.write("Data Preview:")
st.dataframe(df.head())

# 2. Visualization Tab
st.header("Data Visualization")
    
# PCA 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.iloc[:, :-1])
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]

st.subheader('2D PCA Visualization')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pca-one', y='pca-two', hue=df.columns[-2], data=df)
st.pyplot(plt)

# UMAP 3D
reducer = umap.UMAP(n_components=3)
umap_result = reducer.fit_transform(df.iloc[:, :-1])
    
st.subheader('UMAP 3D Visualization')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=df.iloc[:, -1])
st.pyplot(fig)
    
# EDA Plots
st.subheader('Exploratory Data Analysis')
st.write('Distribution of features')
for col in df.columns[:-1]:
	st.write(f'Distribution of {col}')
	sns.histplot(df[col], kde=True)
	st.pyplot(plt)
    
# 3. Feature Selection Tab
st.sidebar.header("Feature Selection")
num_features = st.sidebar.slider("Select number of features to retain", min_value=1, max_value=df.shape[1]-1, value=5)

st.header("Feature Selection using Variance Threshold")
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()
df_reduced = selector.fit_transform(df.iloc[:, :-1])
df_reduced = pd.DataFrame(df_reduced, columns=df.columns[:-1][:num_features])
df_reduced['label'] = df.iloc[:, -1]
st.write(f'Dataset after feature selection with {num_features} features:')
st.dataframe(df_reduced.head())
    
# 4. Classification Tab
st.sidebar.header("Classification")
classifier = st.sidebar.selectbox("Choose classifier", ["KNN", "Random Forest"])
parameter_value = st.sidebar.slider(f"Set parameter for {classifier}", min_value=1, max_value=10, value=3)

st.header(f"Classification with {classifier}")

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(df_reduced.iloc[:, :-1], df_reduced.iloc[:, -1], test_size=0.2, random_state=42)

if classifier == "KNN":
	model = KNeighborsClassifier(n_neighbors=parameter_value)
else:
	model = RandomForestClassifier(n_estimators=parameter_value)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model_reduced = model
model_reduced.fit(X_train_reduced, y_train_reduced)
y_pred_reduced = model_reduced.predict(X_test_reduced)

st.write("Original Data Performance:")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
st.write("ROC-AUC Score:", roc_auc_score(y_test, y_pred, multi_class='ovr'))

st.write("Reduced Data Performance:")
st.write("Accuracy:", accuracy_score(y_test_reduced, y_pred_reduced))
st.write("F1-Score:", f1_score(y_test_reduced, y_pred_reduced, average='weighted'))
st.write("ROC-AUC Score:", roc_auc_score(y_test_reduced, y_pred_reduced, multi_class='ovr'))

# 5. Info Tab
st.sidebar.header("About")
st.sidebar.info("This application is developed as part of a data mining and machine learning project.")


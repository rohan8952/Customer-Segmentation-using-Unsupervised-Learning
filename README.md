Customer Segmentation using Unsupervised Learning

Overview

This project applies unsupervised machine learning (K-Means clustering) to segment customers into distinct groups based on their demographic and behavioral attributes. The goal is to uncover natural patterns in customer data — without any predefined labels — that can be used for targeted marketing, personalized recommendations, and better customer understanding.

Dataset

The dataset (Train.csv) contains 8,068 customer records with the following features:

Column Description ID Unique customer identifier Gender Male / Female Ever_Married Marital status (Yes/No) Age Customer age Graduated Whether the customer graduated (Yes/No) Profession Customer's profession (Artist, Doctor, Engineer, etc.) Work_Experience Years of work experience Spending_Score Spending category (Low / Average / High) Family_Size Number of family members Var_1 Anonymized categorical attribute Segmentation Original label (dropped — not used, since this is an unsupervised task)

Project Workflow

Data Cleaning Dropped the ID column (not useful for clustering) and the original Segmentation label (to keep the task purely unsupervised). Handled missing values: Mode imputation for categorical columns: Ever_Married, Graduated, Profession, Var_1 Median imputation for numerical columns: Work_Experience, Family_Size Removed 947 duplicate rows, reducing the dataset from 8,068 → 7,121 records.
Exploratory Data Analysis (EDA) Plotted distributions of numerical features (Age, Work_Experience, Family_Size) using histograms and boxplots to check spread and outliers. Plotted count plots for categorical features (Gender, Ever_Married, Graduated, Profession, Spending_Score, Var_1) to understand category balance. Generated a correlation heatmap for numerical features to check for multicollinearity.
Feature Engineering One-Hot Encoding applied to categorical columns (Gender, Ever_Married, Graduated, Profession, Spending_Score, Var_1) using OneHotEncoder. Standardization applied to numerical columns (Age, Work_Experience, Family_Size) using StandardScaler, so all features contribute equally to distance-based clustering. Combined encoded and scaled features into a final feature matrix X with 28 features across 7,121 customers.
Choosing the Optimal Number of Clusters
Two standard techniques were used to determine the best value of K:

Elbow Method: Plotted inertia (within-cluster sum of squares) for K = 2 to 10, looking for the "elbow" point where adding more clusters gives diminishing returns. Silhouette Score: Computed for K = 2 to 10 to measure how well-separated and cohesive the clusters are.

Based on both metrics, K = 3 was selected as the optimal number of clusters.

Clustering Applied K-Means (n_clusters=3, random_state=42, n_init=10) to the processed feature matrix. Assigned each customer a cluster label (0, 1, or 2).
Cluster sizes:

Cluster Count % of Total 0 3,361 47.2% 2 2,203 30.9% 1 1,557 21.9% 6. Cluster Profiling

Each cluster was analyzed across all key features using group means, medians, and cross-tabulations:

Cluster 0 — Older, Married, Moderate-to-Low Spenders

Average age: ~55 years Work experience: ~1 year (low) 91% ever married 75% graduated Spending: mostly Average/Low, few High spenders Male-skewed (~59%)

Cluster 1 — Mid-Age, Experienced, Mixed Spenders

Average age: ~38 years Work experience: ~8 years (highest among clusters) 53% ever married 69% graduated Spending: more diverse — meaningful shares across Low/Average/High Roughly gender-balanced

Cluster 2 — Younger, Unmarried, Larger Families, Low Spenders

Average age: ~30 years Work experience: ~1 year (low) Only 17% ever married 38% graduated (lowest) Largest average family size (~3.6) Spending: overwhelmingly Low (92%)

These profiles were validated visually using boxplots (Age, Work_Experience, Family_Size across clusters), count plots (Spending_Score, Marital Status, Graduation Status across clusters), and a heatmap of profession distribution per cluster.

Dimensionality Reduction & Visualization Applied PCA (2 components) to the 28-dimensional feature matrix for visualization purposes. The first two principal components explain ~42% of total variance (PC1: 24.4%, PC2: 17.5%). Plotted a 2D scatter plot of customers colored by cluster, confirming that the three segments are visually distinguishable, with some natural overlap at the boundaries (expected given the moderate variance explained). Tech Stack Python pandas, numpy — data manipulation matplotlib, seaborn — visualization scikit-learn — StandardScaler, OneHotEncoder, KMeans, PCA, silhouette_score Key Takeaways Demonstrates a complete unsupervised learning pipeline: cleaning → EDA → encoding/scaling → optimal K selection → clustering → profiling → dimensionality reduction & visualization. The three discovered segments have clear, interpretable real-world meaning (age/life-stage, spending behavior, family size) despite no labels being used during training. Such segmentation can directly inform business decisions like targeted marketing campaigns, personalized offers, and customer retention strategies.

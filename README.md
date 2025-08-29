# Movie Recommendation System (ALS)

![python](https://img.shields.io/badge/python-3.9%2B-blue)  
![Pandas](https://img.shields.io/badge/Pandas-data-informational)  
![NumPy](https://img.shields.io/badge/NumPy-array-informational)  
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)  
![implicit](https://img.shields.io/badge/implicit-ALS-green)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-plots-informational)

This repository implements a **movie recommendation system** using **ALS (Alternating Least Squares)**.  
User–movie interactions (ratings/watch history) are processed into a sparse user–item matrix.  
The model is trained with **implicit ALS**, and produces both **personalized movie recommendations** and **similar movie lookups**.

## 🧹 Data Preparation
- Use `ratings.csv` containing `userId, movieId, rating`.
- Ratings can be scaled into preference scores (e.g., min-max normalization or log-scaling) for implicit ALS.
- Construct a user–movie interaction matrix with `scipy.sparse.csr_matrix`.

## 🎯 Model (ALS with implicit library)
**Training workflow:**
1. Split data into train/test.  
2. Build the CSR matrix.  
3. Train with `implicit.als.AlternatingLeastSquares`:  
   - `factors` (latent dimension),  
   - `regularization`,  
   - `iterations`,  
   - `alpha` (confidence weight).  
4. Evaluate using ranking metrics: Precision@K, Recall@K, MAP@K.  
5. Generate recommendations:  
   - **For a user**: `model.recommend(user_id, user_items, N=10)`  
   - **Similar movies**: `model.similar_items(movie_id, N=10)`

> Note: For explicit ratings, convert them into implicit preferences (e.g., rating > 3 ⇒ positive feedback).

## 📊 Results
- Save evaluation metrics (Precision@K/Recall@K) and sample recommendation lists in the `results/`.
- Visualizations (matplotlib) and example outputs can also be included here.


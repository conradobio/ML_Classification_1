# Machine Learning - Classify Song Genres from Audio Data

1. BUSINESS PROBLEM

Train a classifier to distinguish between the two genres based only on track information derived from Echonest (now part of Spotify).

Full Machine Learning Project - 

Subsetting the data, aggregating information, and creating plots when exploring the data for obvious trends or factors you should be aware of when doing machine learning.

---

2. BUSINESS ASSUMPTIONS

It is completely unfeasible to classify the type of music individually, it is only possible to do this from some intelligence, in the case of Machine Learning.
With a better rating of songs, it is possible to create a recommendation system for users.
The purpose of this project is only to analyze and classify the songs between 'Rock' and 'Hip-Hop'. However, the same analysis can be extrapolated to any type of musical genre.

---

3. SOLUTION STRATEGY

. Extracting data from an 'Echo Nest' API (the focus of this project is not extracting data from this API).

. Exploratory analysis of two files, one containing the raw data regarding the classification of musical styles and the other containing the songs that will be classified.

. To facilitate the analysis of the model, assess the relationship of the variables with each other to check if there is any variable in the model that has a high correlation.

. If you have to use this variable as main.

. If no main variable is found, then it is necessary to normalize the data.
To reduce the complexity of the model, since no feature with high correlation was found, use the PCA (Principal Component Analysis) method.
In order to be able to apply the PCA, we have to initially normalize the data to prevent the data from being biased in the analysis.

Set to 'features' = all features to define the music style
Defined as 'label' = the music genre.
I will use the 'Standardization' method making the data mean = 0 and standard deviation =1.

```python
# Define our features 
features = echo_tracks.drop(columns=['genre_top', 'track_id'])

# Define our labels
labels = echo_tracks['genre_top']

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
```

It is now possible to use the PCA method to find out how much we can reduce in our data.
Let's plot a graph to be able to visualize it more clearly.

```python
# This is just to make plots appear in the notebook
%matplotlib inline

# Import our plotting module, and PCA class
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
ax.set_xlabel('Principal Component #')
```

Unfortunately with the preview it was not possible to determine the 'elbow' , which means that it is not easy to find the number of dimensions using this method.
The 'cumulative explained variance plot' graph can be used, adopting the 'cut grade' as 85% of the variance.
From the graph it is possible to adopt 5 subsets.
. Algorithm Training
Use DecisionTree and split the variables between training and testing.
. Comparison between the models - DecisionTree and LogisticRegression

![image](https://user-images.githubusercontent.com/72289622/128395056-0b7180a1-5157-49e4-9937-0badfa62c013.png)

When evaluating the data, it was noticed that there was much more 'Rock' data in the dataset, which made the model biased to have better performance than the other musical genre.
. Balance the Dataset to have the same number of data as 'Rock' and 'Hip-Hop'.
. Retrain the algorithms on the 2 models to compare the performance improvement.

![image](https://user-images.githubusercontent.com/72289622/128395099-89bae808-0c9f-4906-ad4d-02a6410a3d0c.png)

It is possible to see an incredible improvement in the model only with the basis of balancing the Datasets.
. Use the 'cross-validation' method to evaluate the models with cv=10 and the 'K-fold' method which consists of splitting the data into different K of the same size subset and then iteratively using each subset as a test set while using the rest of the data as train sets.
The final result of the model was (the average of 10 values was taken)

![image](https://user-images.githubusercontent.com/72289622/128395118-8dd0da24-68c4-403c-b179-71ffb16bbf94.png)

---

4. TOP 3 insights

.. The data was balanced, since the amount of 'Rock' data was much greater than 'Hip Hop', generating a biased model and when balancing the model, the performance of the algorithms was adjusted.

---

5. Machine Learnin Model Aplicados

DecisionTree, LogisticRegression

To normalization- Standardization and PCA.

---

6. Machine Learning Performance


![image](https://user-images.githubusercontent.com/72289622/128395157-3793969a-ec3d-4ca5-9a9d-463084c3ffc3.png)

---

7. Business Results

With the increased accuracy of the algorithms, it is possible to improve the music recommendation system for users and thereby increase user engagement with the platform, thus reducing churn (due to lack of use of the platform) and reducing users' time stay on the platform.

---

8. Conclusion

---

9. Lesson Learned

---

10. Next Steps to Improve
Apply model to other music genres

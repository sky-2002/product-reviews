## Product reviews sentiment analysis [WIP]

This is a sentiment analysis project created using a product reviews dataset given [here](https://gfc.target.com/cmsti/f46a7ece-a8d2-4c71-a5ef-1cc9c3df8e3e.xlsx). It includes product information and ratings and reviews given by users. I have assumed review score less than 3 to be negative and greater than 3 to be positive on a scale of 5. This makes it an imbalanced dataset.

### Udates 
- Feb 10, 2024
    - I have added summarization option for reviews using `LaMini-Flan-T5-248M` model. Now we can see summaries of long reviews to extract crisp information and get better insight from summary itself.
    This would make it easier to understand the issues without the need to scroll through entire review.

### Data analysis

I have deployed a [streamlit app](https://reviewsentimentaakash.streamlit.app/) so that anyone can perform the analysis by selecting a product category.

I did a thorough data analysis to understand the underlying patterns so as to take decisions while modelling. Below are some plots which make it easier to understand this data.

Rating Distribution | Review length distribution | Review length by rating boxplot
--------------------|----------------------------|--------------------------------
![Rating distribution](./images/rating_dist.png) | ![Review length](./images/review_length_dist.png) | ![Review length by rating](./images/word_count_boxplot.png)

Based on the above rating distribution, I had made an assumption that rating less than equal to 3 will be considered a negative review. And hence the other two plots.

### Sentiment Analysis
#### <u>Modelling</u>

Initially, I have resorted to `sklearn` models as they are easier to interpret. I started with simple feature extraction methods and built upon the results to improve.

I got very good overall F1-score results with SVMs, Random Forests, Boosting classifiers etc. But the flaw in them was that they performed poorly on the minority class. 
Below are the confusion matrices.

Linear SVM | Random Forest | Linear SVM with SMOTE
-----------|---------------|------------------
![Linear SVM TF](./images/tf_linear_svm_cm.png) | ![RF TF](./images/tf_rf_cm.png) | ![Linear SVM SMOTE](./images/smote_linear_svm_cm.png)

We can see from the confusion matrix that metrics(expecially recall) on the `negative` class are not that good for the first two models. But we can see an improvement in recall for the model which used SMOTE augmented features.

**To handle class imbalance, I then used SMOTE, to perform oversampling on the term frequency vectors for the minority class and undersampling of the majority class(see third figure).**

But using term frequency or TF-IDF was not sufficient, as the results were still not as good as can be used for a real world use case. 

#### <u>Improving performance</u>

I considered using `sentence-transformer` models to get embeddings for reviews and then use these embeddings as features to `sklearn` models.
I experimented with two models - `all-MiniLM-L6-v2` and `nomic-embed-text-v1`. Used LanceDB to compute and store these embeddings.

Below are the UMAP reduced 2D representations of the review embeddings using these two models.

all-MiniLM-L6-v2 | nomic-embed-text-v1
-----------------|--------------------
![allmini](./images/all_mini_lm_l6_v2_all_2d.png) | ![nomic](./images/nomic_all_2d.png)

I have then used the SMOTE resampled embeddings using these two models and passed them to a linear SVM, the results are impressive.

Linear SVM with all-Mini embeddings | Linear SVM with Nomic embeddings
------------------------------------|----------------------------------
![all-mini](./images/linear_svm_smote_allmini.png) | ![nomic](./images/linear_svm_nomic_cm.png)


#### <u>Comparison of model performance with pre-trained models</u>

Nomic embedding and SMOTE based Linear SVM | Flair Sentiment analysis pre-trained model
-------------------------------------------|-------------------------------------------
![nomic smote perf](./images/nomic_smote_perf.png) | ![flair perf](./images/flair_perf.png)

This comparison shows that our linear SVM is at par with the pre-trained Flair model, which is based on distilBERT. In fact, our model has **`8% more recall`** on the `negative` class than the pre-trained model. 

Note: Inference using the Flair model was performed on Google colab with T4 GPU.

These are the model performances, I did many experiments while creating this repo, putting some of them here.

Model | Accuracy | Precision | Recall | F1-Score
------|----------|-----------|--------|----------
MultinomialNB | 0.9377  | 0.9241 | 0.9377 | 0.9261
Random_Forest | 0.9331 | 0.9376 | 0.9331 | 0.9017
Linear_SVM_SMOTE_Embeddings | 0.8024 | 0.9228  | 0.8024 | 08464
Linear_SVM_Nomic_SMOTE | 0.8952 | 0.9402 | 0.8952 | 0.9114

### Note: This is a work in progress(hence the WIP). I will keep updating this repository as I do more analysis.

### Planned features
- A better UI
    - Needs to be smooth and good-looking. Need some streamlit and CSS hacks.
    - Need better organization of the figures and elements.
- Query product review
    - I am planning to add a method to get all useful comments for a particular product by querying all the reviews for that product. 
    - This would help understand the reviews better than just summarizing them. For example, if I ask "How is the after-sales service for this product?", this method should give me all those reviews(or their summary) that talk about "after-sales service".
- Make the streamlit app modular
    - I had started with limited features and hence wrote everything in the `app.py` file. Need to shift methods to files and modules.
    - Organize modules into ones those transform data, visualizers, AI related etc.


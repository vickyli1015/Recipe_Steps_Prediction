<!-- #region -->
# Spice Up Your Models: Decoding Recipe Step Predictions
Predicting the Number of Steps in a Recipe Using Models

by Vicky Li: <yil164@ucsd.edu>

Our exploratory data analysis on this dataset can be found [here](https://vickyli1015.github.io/Recipes-Rating-Analysis/).

## Framing the Problem

### Prediction problem of interest: 

Predict the **number of steps in recipes** using features from the combined data frame containing recipes and ratings.

Since number of steps is a quantitative variable, this is a **regression problem**. I'm going to use a **regression model** to predict it, i.e., the ***n_steps* column** specifically. I chose it for personal reasons. I am a lazy person and I don't want to spend so much time on cooking/preparing for cooking, so I am so curious about what characteristics of a recipe may contribute to number of steps involved!

For this question, I am going to use **RMSE** as the metric to evaluate the model. It comes with the **same unit of the response variable (n_steps)**, making it easier to interpret with context. For example, if RMSE = 3, that means the **average magnitude of the residuals**, i.e, the mean error that this model has for predicting the number of steps involved is around 3 **steps**! Also, it is useful since we want to prioritize reducing larger deviations, which is also why I prefer RMSE more than metrics like R-sqaured. Since our model is not a classifier, metrics like F1-score and precision does not work here.

## Baseline Model

(Note: the statistics mentioned below may be different in the notebook since I rerun the whole notebook before submission, but you the typically follow the same trend described below)

Given the dataset containing recipe and ratings information as shown below: 

(notice that for some columns that contains long strings/list of long strings, only the first few words/items are included for readability)

|   | name                                 | id     | minutes | contributor_id | submitted  | tags                                                  | nutrition                                     | n_steps | steps                                        | description                            | ingredients                                           | n_ingredients | average_rating |
|---|--------------------------------------|--------|---------|----------------|------------|-------------------------------------------------------|-----------------------------------------------|---------|----------------------------------------------|----------------------------------------|-------------------------------------------------------|---------------|----------------|
| 0 | 1 brownies in the world    best ever | 333281 | 40      | 985201         | 2008-10-27 | ['60-minutes-or-less', 'time-to-make', 'course',...]  | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]      | 10      | ['heat the oven to 350f and...'...]          | these are the most; chocolatey,...     | ['bittersweet chocolate', 'unsalted butter'...]       | 9             | 4.0            |
| 1 | 1 in canada chocolate chip cookies   | 453467 | 45      | 1848091        | 2011-04-11 | ['60-minutes-or-less', 'time-to-make', 'cuisine'...]  | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0]  | 12      | ['pre-heat oven the 350...'...]              | this is the recipe that we use...      | ['white sugar', 'brown sugar', 'salt',...]            | 11            | 5.0            |
| 2 | 412 broccoli casserole               | 306168 | 40      | 50969          | 2008-05-30 | ['60-minutes-or-less', 'time-to-make', 'course', ...] | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]     | 6       | ['preheat oven to 350 degrees...'...]        | since there are already 411 recipes... | ['frozen broccoli cuts', 'cream of chicken soup',...] | 9             | 5.0            |
| 3 | millionaire pound cake               | 286009 | 120     | 461724         | 2008-02-12 | ['time-to-make', 'course', 'cuisine', ...]            | [878.3, 63.0, 326.0, 13.0, 20.0, 123.0, 39.0] | 7       | ['freheat the oven to 300 degrees...'...]    | why a millionaire pound cake?...       | ['butter', 'sugar', 'eggs',...]                       | 7             | 5.0            |
| 4 | 2000 meatloaf                        | 475785 | 90      | 2202916        | 2012-03-06 | ['time-to-make', 'course', 'main-ingredient', ...]    | [267.0, 30.0, 12.0, 12.0, 29.0, 48.0, 2.0]    | 17      | ['pan fry bacon , and set aside on a...'...] | ready, set, cook! special edition...   | ['meatloaf mixture', 'unsmoked bacon',...]            | 13            | 5.0            |

I am going to start off by predicting the number of steps (n_steps) in a recipe using **minutes (minutes)** and the **number of ingredients (n_ingredients)** of the recipe. In this case, I am going to use a **linear regression model** to get started. Linear regression model tries to capture the relationship between features and the response variable by **fitting a linear line** through the data points. Both features are quantitative, and I will leave them as-is (though I still use a FunctionTransformer to keep them as-is) when training the pipeline model to. However, this should not affect the model’s natural ability to predict.

The model has been tested using **train-test-split** to evaluate the model's ability to generalize to predict unseen data well. The training set has a RMSE of 5.75404326418339 while the test set has a RMSE of 5.810676060245601 (This may not be accurate but the range of difference is). Since the model is trained on the training set, it is usual that the training set achieves a **lower** RMSE, but if the RMSE is **much larger** for the test set than the training set, it may be an evidence of overfitting. However, the difference typically varies between 0.03 - 0.2 steps in our case, which is not that significant. Therefore, the model is **not overfitting**.

However, consider the mean and quantiles of the number of steps (shown below), 

| Statistic Type | Statistic  Value |
|-------|--------------|
| count | 83782.000000 |
|  mean | 10.105440    |
| std   | 6.390109     |
| min   | 1.000000     |
| 25%   | 6.000000     |
| 50%   | 9.000000     |
| 75%   | 13.000000    |
| max   | 100.000000   |

apparently most of the recipes have a number of steps <= 13, which means a RMSE of around 5 steps is actually pretty high. Therefore, it may be an evidence of **underfitting**, i.e., the model is not complex enough to predict the number of steps with a higher accuracy so that it results in a lower RMSE for even only the training set.

Since the model is underfitting that much that it mis-predicts 1/2 of the mean number of steps, I would say that **the model is not good enough** and there is absolutely a better model out there.

## Final Model

For the final model, I started off by transforming the features present in the baseline model. 

For *minutes*: 

```py
recipes['minutes'].describe()
```

| Statistic Type | Statistic  Value |
|-------|--------------|
| count | 8.378200e+04 |
| mean  | 1.150309e+02 |
| std   | 3.990871e+03 |
| min   | 0.000000e+00 |
| 25%   | 2.000000e+01 |
| 50%   | 3.500000e+01 |
| 75%   | 6.500000e+01 |
| max   | 1.051200e+06 |


considering that ***minutes* span from one digit to a million**, I wanted to make the numbers smaller/more concentrated, especially the outliters, which evidently makes mean to be even larger than 75th percentile (75%) here. **Outliers may have a significant impact on the linear regression model and it needs to be addressed!** a **QuantileTransformer** is robust to outliers. 

- It also preserves the relative **order and distribution** of the *minutes* values, while also provides a straight-forward interpretation of where each *minutes* value sits in the overall *minutes* values.
- It 

Overall, I believe that this transformer helps reduce the noise from the outliers.

For *n_ingredients* column:

```py
recipes['n_ingredients'].describe()
```

| Value | Count |
|----------------|-----------------|
| 5.000000       | 47784           |
| 4.000000       | 12217           |
| 4.500000       | 4821            |
| 3.000000       | 2508            |
| 4.666667       | 2265            |
| 4.750000       | 1283            |
| 4.333333       | 1170            |
| 4.800000       | 858             |
| 3.500000       | 662             |
| 2.000000       | 618             |

you can see that the distribution for the number of ingredients is still skewed here, since mean > median (50%), however it is much better than the *minutes* column. 
- Still, we want to address outliers! a **StandardScaler** reduces the effect of the outliers by **centering all values around zero and having a unit variance**, which allows the linear regression model (and regression models in general) to assign a more appropriate weight and attention to each feature value.
- Using a StandardScaler **makes the regression coefficients more meaningful** as they represent the change in the number of steps (y) associated with a one-standard-deviation change in the 'n_ingredients' feature

Overall, I believe that this transformer helps reduce the noise from the outliers, while also ensuring the transformed values help the linear regression models to perform better when estimating the intercept and coefficients.

After applying those transformers, we get (train_set_RMSE, test_set_RMSE) == (5.52111626057998, 5.581885926716189). 
- The difference in RMSE between the training set and the test set has decreased/is roughly the same, which means **the model predicts better/the same on unseen data**, but overall they are still high in the sense that 5 steps are 1/2 of the average steps we typically see! 
- Therefore, **the Linear Regression Model may be essentially not suitable for this predition**, such that feature transforming does not help. Reasons for that may be the relationship between the features and the number of steps (n_steps) is not linear, and the fact that there is not really a lot of options of hyperparameters to choose from (if any).

--- 
To combat those limitations, **I will use a Decision Tree Regressor as the final model** since it can **also** capture **non-linear relationships** between features and n_steps and I will be able to find and use the **best hyperparameter(s)** to prevent one of its disadvantages -- overfitting/not general enough, to some extent.

The hyperparameters I plan to tune are **"max_depth"** and **"min_samples_split"**.
- we want to have the best "max_depth" so that the model will not underfit or overfit. It underfits when there is not a sufficient amount of qustions being asked, i.e., the depth of the tree is not deep enough. It overfits when there are too many questions being asked, whose answers were **memorized and considered to be the correct response values** such that the questions will separate the data points in unexpected complicated (not general) way.

- We want to have the best min_samples_split. min_samples_split helps control the complexity of the decision tree model since it **specifies the minimum number of samples required to split an internal node**. The larger it is, it is more general, but it can underfit. The smaller it is, the more complicated it is, so it can overfit. Therefore, it is so important to find a balance here! Also, the best min_samples_split should make the model general enough that **there is a concrete set of rules that predicts their relationships*.

By using **GridSearchCV** over a list of hyperparameter options (below),

```py
>>> hyperparameter_list = {
    'max_depth': [2, 3, 4, 5, 7, 10, 13, 15, 18, 20, None], 
    'min_samples_split': [2, 5, 10, 20, 50, 100, 200]
    }
>>> searcher = GridSearchCV(DecisionTreeRegressor(), hp, cv=5)
>>> searcher.fit(X_train, y_train)
>>> best = searcher.best_params_
>>> best
{'max_depth': 10, 'min_samples_split': 200}
```
we found that the best hyperparameter for "max_depth" is 10, and 200 for "min_samples_split".

By training the model using the optimal hyperparameters, (train_set_RMSE, test_set_RMSE) reduces to (5.374887598821211, 5.492137521112). 

Compared to the baseline model, the **RMSE overall has decreased by around 0.4-0.5 (steps)**, from which we can see that the model performance has improved by some extent by using a new regression model and doing feature transformations that address for effects of outliers. The difference between RMSE for the training set and the test set has increased, but even though the problem of **overfitting, which is one of the natural disadvantages of a Decision Tree, is moderately more explicit, the model does optimize accuracy overall!**


## Fairness Analysis

By exploring the frequency of some possible values for *average rating*, we found that there are high rating values ((average rating >= 4) with high frequency, but there are also high rating values that has low frequency. The same applies for lower ratings values (average rating < 4)

```py
recipes['average_rating'].value_counts().iloc[:10]
```

| Value    | Count |
|----------|-------|
| 5.000000 | 47784 |
| 4.000000 | 12217 |
| 4.500000 | 4821  |
| 3.000000 | 2508  |
| 4.666667 | 2265  |
| 4.750000 | 1283  |
| 4.333333 | 1170  |
| 4.800000 | 858   |
| 3.500000 | 662   |
| 2.000000 | 618   |


```py
recipes['average_rating'].value_counts().iloc[-10:]
```

| Value    | Count |
|----------|-------|
| 4.280000 | 1     |
| 4.628571 | 1     |
| 4.382353 | 1     |
| 4.656250 | 1     |
| 4.638298 | 1     |
| 4.584906 | 1     |
| 4.805556 | 1     |
| 4.238095 | 1     |
| 2.800000 | 1     |
| 4.541667 | 1     |

```py
#There are 402 unique average rating values!
>>> print(pd.Series(recipes['average_rating'].unique()).shape[0])
402
```

There are a lot of possibilities for average ratings, and the frequency of each differs a lot! Let's see if those factors make a difference for predictions of number of steps!

**Group X**: recipes that has average rating >= 4 (high-rating recipes)

**Group Y**: recipes that has average rating < 4 (not high-rating recipes)

**Evaluation Metric**: since we care about how far off the predictions of number of steps are from the actual number of steps for recipes in both groups, we use RMSE.

#### Hypothesis Set-Up:

- **Null Hypothesis**: Our model is fair. Its RMSE for high-rating recipes and not high-rating recipes are roughly the same, and any differences are due to random chance.
- **Alternative Hypothesis**: Our model is unfair. Its RMSE for high-rating recipes is lower than its RMSE for not high-rating recipes. i.e., the model is more likely to predict the n_steps of high-rating recipes with smaller errors (RMSE).

**Test Statistic**: difference between the RMSE of high-rating recipes and the RMSE of the recipes with not so high ratings. (RMSE_high_rating_recipes - RMSE_low_rating_recipes)

**Significance Level**: 5%

#### Permutaion Test
During each permutation iteration out of the 1000 permutations:
- I shuffle the average ratings randomly. By doing so, each recipe may now have a high/low average rating instead of being fixed.
- I **predict the number of steps using the Final Model with features (minutes and number of ingredients) of recipes with high ratings and recipes with low ratings separately**. 
- Lastly, I calculate their RMSE and their differences (test statistic).

Finally, I compare the list of test statistics with the observed statistic derived from our original dataset to calculate the p-value.

**p-value:**

```py
>>> p_value = (np.array(stats) < observed_stat).mean()
>>> p_value
0.175
```

Here is a plot of the distribution of test statistics.

<iframe src="'/Users/vickyli/Dropbox/My Mac (Vicky的MacBook Air)/Desktop/Recipe_Steps_Prediction/data_vis/fairness_permutaion.html'" width=800 height=600 frameBorder=0></iframe>

**Conclusion**: Since the p-value is > 0.05, and that the observed statistic is not very far off from many of the test statistics, we **fail to reject** that the Final Model is fair when it predicts number of steps for recipes with a high rating and a not so high rating, with a significance level of 5%.
<!-- #endregion -->

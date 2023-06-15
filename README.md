# Spice Up Your Models: Decoding Recipe Step Predictions
Predicting the Number of Steps in a Recipe Using Models
by Vicky Li: <yil164@ucsd.edu>

Our exploratory data analysis on this dataset can be found [here](https://vickyli1015.github.io/Recipes-Rating-Analysis/)

## Framing the Problem

### Prediction problem of interest: 

Predict the **number of steps in recipes** using features from the combined data frame containing recipes and ratings.

Since number of steps is a quantitative variable, this is a **regression problem**. I'm going to use a **regression model** to predict it, i.e., the ***n_steps* column** specifically. I chose it for personal reasons. I am a lazy person and I don't want to spend so much time on cooking/preparing for cooking, so I am so curious about what characteristics of a recipe may contribute to number of steps involved!

For this question, I am going to use **RMSE** as the metric to evaluate the model. It comes with the **same unit of the response variable (n_steps)**, making it easier to interpret with context. For example, if RMSE = 3, that means the **average magnitude of the residuals**, i.e, the mean error that this model has for predicting the number of steps involved is around 3 **steps**! Also, it is useful since we want to prioritize reducing larger deviations, which is also why I prefer RMSE more than metrics like R-sqaured. Since our model is not a classifier, metrics like F1-score and precision does not work here.

## Baseline Model

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

| count | 83782.000000 |
|-------|--------------|
| mean  | 10.105440    |
| std   | 6.390109     |
| min   | 1.000000     |
| 25%   | 6.000000     |
| 50%   | 9.000000     |
| 75%   | 13.000000    |
| max   | 100.000000   |

apparently most of the recipes have a number of steps <= 13, which means a RMSE of around 5 steps is actually pretty high. Therefore, it may be an evidence of **underfitting**, i.e., the model is not complex enough to predict the number of steps with a higher accuracy so that it results in a lower RMSE for even only the training set.

Since the model is underfitting that much that it mis-predicts 1/2 of the mean number of steps, I would say that the model is not good enough and there is absolutely a better model out there.

## Final Model

## Fairness Analysis

```python

```

import math
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

sns.set()

# Ask the user for the CSV file name
csv_file = input("Please enter the CSV file name (with extension): ")

products = pd.read_csv(csv_file)
np.random.seed(416)
products = products[products['rating'] != 3].copy()
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

def remove_punctuation(text):
    if type(text) is str:
        return text.translate(str.maketrans('', '', string.punctuation))
    else:
        return ''

products['review_clean'] = products['review'].apply(remove_punctuation)
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(products['review_clean'])
features = vectorizer.get_feature_names_out()
product_data = pd.DataFrame(count_matrix.toarray(), index=products.index, columns=features)
product_data['sentiment'] = products['sentiment']
product_data['review_clean'] = products['review_clean']
product_data['summary'] = products['summary']
train_data, test_and_validation_data = train_test_split(product_data, test_size=0.2, random_state=3)
validation_data, test_data = train_test_split(test_and_validation_data, test_size=0.5, random_state=3)

if product_data[product_data["sentiment"] == 1].shape[0] >= product_data[product_data["sentiment"] == -1].shape[0]:
    majority_label = 1
else:
    majority_label = -1

majority_classifier_validation_accuracy = (validation_data[validation_data["sentiment"] == majority_label].shape[0]) / (validation_data.shape[0])

sentiment_model = LogisticRegression(penalty='l2', C=1e23, random_state=1)
sentiment_model.fit(train_data[features], train_data['sentiment'])

coefficients = sentiment_model.coef_
featurenames = sentiment_model.feature_names_in_
most_negative_word = featurenames[np.argmin(coefficients)]
most_positive_word = featurenames[np.argmax(coefficients)]

print(f"Most Negative Word: {most_negative_word}")
print(f"Most Positive Word: {most_positive_word}")

sample_data = validation_data[8:11]
predicted_probabilities = sentiment_model.predict_proba(sample_data[features])
predicted_labels = sentiment_model.predict(sample_data[features])

print("Predicted Probabilities for Sample Data:")
print(predicted_probabilities)
print("Predicted Labels for Sample Data:")
print(predicted_labels)

predictions = sentiment_model.predict_proba(validation_data[features])
negatives = np.array([x[0] for x in predictions])
positives = np.array([x[1] for x in predictions])
max_negative = np.max(negatives)
max_positive = np.max(positives)
most_negative_review = validation_data.iloc[np.where(negatives == max_negative)[0][0]]['review_clean']
most_positive_review = validation_data.iloc[np.where(positives == max_positive)[0][0]]['review_clean']

print(f"Most Positive Review: {most_positive_review}")
print(f"Most Negative Review: {most_negative_review}")

predictedsentiments = sentiment_model.predict(validation_data[features])
sentiment_model_validation_accuracy = accuracy_score(validation_data["sentiment"], predictedsentiments)

print(f"Sentiment Model Validation Accuracy: {sentiment_model_validation_accuracy}")

validation_data["predictions"] = sentiment_model.predict(validation_data[features])
tp = validation_data[(validation_data["predictions"] == 1) & (validation_data["sentiment"] == 1)].shape[0]
fp = validation_data[(validation_data["predictions"] == 1) & (validation_data["sentiment"] == -1)].shape[0]
tn = validation_data[(validation_data["predictions"] == -1) & (validation_data["sentiment"] == -1)].shape[0]
fn = validation_data[(validation_data["predictions"] == -1) & (validation_data["sentiment"] == 1)].shape[0]

print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")

def plot_confusion_matrix(tp, fp, fn, tn):
    data = np.matrix([[tp, fp], [fn, tn]])
    sns.heatmap(data, annot=True, xticklabels=['Actual Pos', 'Actual Neg'], yticklabels=['Pred. Pos', 'Pred. Neg'])
    plt.show()

plot_confusion_matrix(tp=tp, fp=fp, tn=tn, fn=fn)

l2_penalties = [0.01, 1, 4, 10, 1e2, 1e3, 1e5]
l2_penalty_names = [f'coefficients [L2={l2_penalty:.0e}]' for l2_penalty in l2_penalties]
coef_table = pd.DataFrame(columns=['word'] + l2_penalty_names)
coef_table['word'] = features
accuracy_data = []

for l2_penalty, l2_penalty_column_name in zip(l2_penalties, l2_penalty_names):
    newmodel = LogisticRegression(C=1/l2_penalty, fit_intercept=False, random_state=1)
    newmodel.fit(train_data[features], train_data["sentiment"])
    coef_table[l2_penalty_column_name] = newmodel.coef_[0]
    accuracy_data.append({
        "l2_penalty": l2_penalty,
        "train_accuracy": accuracy_score(train_data["sentiment"], newmodel.predict(train_data[features])),
        "validation_accuracy": accuracy_score(validation_data["sentiment"], newmodel.predict(validation_data[features]))
    })

accuracies_table = pd.DataFrame(accuracy_data)

print("Coefficients Table:")
print(coef_table)
print("Accuracies Table:")
print(accuracies_table)

l2pencolumn = 'coefficients [L2=1e+00]'
positive_words = pd.Series(coef_table.nlargest(5, l2pencolumn)['word'])
negative_words = pd.Series(coef_table.nsmallest(5, l2pencolumn)['word'])

print("Top 5 Positive Words:")
print(positive_words)
print("Top 5 Negative Words:")
print(negative_words)

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    def get_cmap_value(cmap, i, total_words):
        return cmap(0.8 * ((i + 1) / (total_words * 1.2) + 0.15))

    def plot_coeffs_for_words(ax, words, cmap):
        words_df = table[table['word'].isin(words)]
        words_df = words_df.reset_index(drop=True)

        for i, row in words_df.iterrows():
            color = get_cmap_value(cmap, i, len(words))
            ax.plot(xx, row[row.index != 'word'], '-', label=row['word'], linewidth=4.0, color=color)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    xx = l2_penalty_list
    ax.plot(xx, [0.] * len(xx), '--', linewidth=1, color='k')

    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    plot_coeffs_for_words(ax, positive_words, cmap_positive)
    plot_coeffs_for_words(ax, negative_words, cmap_negative)

    ax.legend(loc='best', ncol=2, prop={'size': 16}, columnspacing=0.5)
    ax.set_title('Coefficient path')
    ax.set_xlabel('L2 penalty ($\lambda$)')
    ax.set_ylabel('Coefficient value')
    ax.set_xscale('log')

make_coefficient_plot(coef_table, positive_words, negative_words, l2_penalty_list=l2_penalties)

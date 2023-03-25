import math
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# hyperparameters
training_iterations = 7
label_factor = 0.3
n_estimators = 100

def label_highly_confident():
    print("Labeling highly confident data...")
    df = pd.read_csv("./processed_reviews.csv")
    
    # create new series for fraud, initializing all as nan
    df["fraud"] = np.nan
    
    # label highly confident frauds - helpful percentage < 0.3 and helpful total > 10
    df.loc[(df["helpful_percentage"] < 0.2) & (df["helpful_total"] > 10), "fraud"] = 1
    
    # label highly confident non-frauds - helpful percentage > 0.7 and helpful total > 10
    df.loc[(df["helpful_percentage"] > 0.8) & (df["helpful_total"] > 10), "fraud"] = 0
    
    
    df.to_csv("./processed_reviews_with_confidence_labels.csv", index=False)


def load_data():
    print("Loading data...")
    x_labeled, y_labeled, x_unlabeled, x_val, y_val = None, None, None, None, None

    df = pd.read_csv("./processed_reviews_with_confidence_labels.csv")
    df.drop(columns=["reviewerID", "asin", "helpful_percentage"], inplace=True)

    # split data into labeled and unlabeled
    labeled_df = df[df["fraud"].notna()]
    unlabeled_df = df[df["fraud"].isna()]

    # get numpy arrays
    x_labeled = labeled_df.to_numpy()
    x_unlabeled = unlabeled_df.to_numpy()

    # split labeled data into training and validation
    x_labeled, x_val = train_test_split(x_labeled, test_size=0.2, random_state=23)

    # split x and y (last column is fraud = y)
    y_labeled = x_labeled[:, -1]
    y_val = x_val[:, -1]

    # remove fraud column from x_labeled, x_unlabeled, and x_val
    x_labeled = np.delete(x_labeled, -1, axis=1)
    x_unlabeled = np.delete(x_unlabeled, -1, axis=1)
    x_val = np.delete(x_val, -1, axis=1)

    return x_labeled, y_labeled, x_unlabeled, x_val, y_val

def save_labeled_data(x_labeled, y_labeled):
    print("Saving labeled data...")

    xdf = pd.DataFrame(x_labeled)
    
    dictForDf = {
        "X": x_labeled,
        "Y": y_labeled
    }
    dfToSave = pd.DataFrame(dictForDf)
    dfToSave.to_csv("labeled_data.csv")

def main():
    label_highly_confident()

    x_labeled, y_labeled, x_unlabeled, x_val, y_val = load_data()
    rf = RandomForestClassifier(n_estimators=n_estimators)
    
    for i in range(training_iterations):
        print("Training iteration {}/{}...".format(i+1, training_iterations))

        # fit on the labeled data (3 is so we skip reviewId, asin, and helpful_percentage)
        rf.fit(x_labeled[:,3:], y_labeled)

        # then we get the predictions and probabilities for the unlabeled data
        y_unlabeled_pred = rf.predict(x_unlabeled[:,3:])
        y_unlabeled_prob = rf.predict_proba(x_unlabeled[:,3:])[:, 1]

        # we sort the unlabeled data by the probability of being a fraud
        k = len(y_unlabeled_pred) * label_factor
        s = np.argsort(-y_unlabeled_prob)

        # we then take the top k and label them
        pseudolabeled_idx = s[:math.floor(k)]
        unlabeled_idx = s[math.floor(k):]

        x_pseudolabeled = x_unlabeled[pseudolabeled_idx]
        y_pseudolabeled = y_unlabeled_pred[pseudolabeled_idx]
        x_unlabeled = x_unlabeled[unlabeled_idx]

        x_labeled = np.concatenate([x_labeled, x_pseudolabeled])
        y_labeled = np.concatenate([y_labeled, y_pseudolabeled])

        y_val_pred = rf.predict(x_val[:,3:])
        accuracy = accuracy_score(y_val, y_val_pred)
        
        print("\nIteration {}/{}: Validation accuracy: {:.4f}\n".format(i+1, training_iterations, accuracy))
    
    x_pseudolabeled = x_unlabeled
    y_pseudolabeled = rf.predict(x_unlabeled[:,3:])
    x_labeled = np.concatenate([x_labeled, x_pseudolabeled])
    y_labeled = np.concatenate([y_labeled, y_pseudolabeled])

    save_labeled_data(x_labeled, y_labeled)
    print("\nfinished and saved labeled data")


if __name__ == "__main__":
    main()

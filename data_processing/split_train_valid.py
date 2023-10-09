import pandas as pd


# construct a row with sampled candidates
def sample_candidates(row, df):
    sample = df.drop(df[df["left"] == row["left"]].index).sample(19, ignore_index=True)["right"].transpose()
    sample = sample.set_axis([
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "c9",
        "c10",
        "c11",
        "c12",
        "c13",
        "c14",
        "c15",
        "c16",
        "c17",
        "c18",
        "c19",
    ])
    return pd.concat([row, sample])


# training data set variables
validate_ratio = 0.1
training_set = 2000
validate_size = int(training_set * validate_ratio)

original_train = pd.read_csv("train.csv")

# split into training and validation set
valid = original_train.sample(validate_size)
train = original_train.drop(valid.index)

# sample candidates for validation
valid = valid.apply(lambda row: sample_candidates(row, valid), axis=1)

# write to file
train.to_csv("split_train.csv", index=False)
valid.to_csv("split_valid.csv", index=False)

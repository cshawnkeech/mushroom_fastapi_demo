"""
mushroom_model

Functions for predicting whether a given mushroom is poisonous or not.
"""
import joblib
import pandas as pd

PATH = "models/artifacts.joblib"
ARTIFACT = joblib.load(PATH)
MODEL = ARTIFACT['lr_model']
PREPROCESSOR = ARTIFACT['preprocessor']


def predict_mushroom(df):
    """
    Returns ndarray of predictions (poisonous=1) given a properly
    formatted DataFrame of observations.
    """
    # process input df
    processed = PREPROCESSOR.transform(df)

    # return prediction
    return MODEL.predict(processed)


if __name__ == "__main__":

    # confirm ARTIFACT dictionary
    print(
        type(ARTIFACT),
        ARTIFACT.keys(),
        sep="\n")

    observation = {
        'cap-diameter': [50],
        'stem-height': [20],
        'stem-width': [30],
        'has-ring': ['t'],
        'cap-shape': ['c']
    }

    single_obs_df = pd.DataFrame(observation)

    print(predict_mushroom(single_obs_df),
          type(predict_mushroom(single_obs_df)))

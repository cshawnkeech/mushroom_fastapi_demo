"""
FastAPI Mushroom Model API
"""
from joblib import load
import pandas as pd
from fastapi import FastAPI


PATH = "models/artifacts.joblib"
ARTIFACT = load(PATH)
MODEL = ARTIFACT['lr_model']
PREPROCESSOR = ARTIFACT['preprocessor']

app = FastAPI()


def format_and_convert_input(
        cap_diameter: float,
        cap_shape: str,
        has_ring: bool,
        stem_height: float,
        stem_width: float):
    """
    Returns df with single observation ready for model
    """

    cap_shape_map = {
        'conical': 'c',
        'bell': 'b',
        'convex': 'x',
        'flat': 'f',
        'sunken': 's',
        'spherical': 'p',
        'others': 'o'
    }

    has_ring = 't' if has_ring else 'f'

    # return DataFrame with single observation
    return pd.DataFrame({
        "cap-diameter": [cap_diameter],
        "cap-shape": [cap_shape_map[cap_shape]],
        "has-ring": [has_ring],
        "stem-height": [stem_height],
        "stem-width": [stem_width]
    })


def get_prediction(df, model=MODEL, preprocessor=PREPROCESSOR):
    """
    Return a prediction from a preformatted df Returns 1 if poisonous, 0
    if edible NOT FOR REAL MUSHROOMS!
    """

    pred = model.predict(preprocessor.transform(df))

    return {'prediction': int(pred[0])}


@app.get("/")
async def root():
    """return hello world"""

    return {"message": "Hello World"}


@app.get("/mushroom_query")
async def mushroom_query(
        cap_diameter: float,
        cap_shape: str,
        has_ring: bool,
        stem_height: float,
        stem_width: float):
    """Take in query parameters for mushroom model"""

    obs = format_and_convert_input(
        cap_diameter,
        cap_shape,
        has_ring,
        stem_height,
        stem_width)

    return get_prediction(obs)


@app.post("/mushroom_post")
async def mushroom_post(mushroom: dict):
    """Take in json packet for mushroom model"""

    obs = format_and_convert_input(
        mushroom["cap_diameter"],
        mushroom["cap_shape"],
        mushroom["has_ring"],
        mushroom["stem_height"],
        mushroom["stem_width"],
    )

    print(mushroom)
    print(obs)

    return get_prediction(obs)

if __name__ == "__main__":

    TO_DO = """
    * Final Naive MVP
    """

"""
FastAPI Mushroom Model API
"""
from enum import Enum
import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# import constants from another file
from src.restore_artifacts import PREPROCESSOR, MODEL


app = FastAPI()


class CapShape(str, Enum):
    """Enumerate cap_shape"""
    CONICAL = 'conical'
    BELL = 'bell'
    CONVEX = 'convex'
    FLAT = 'flat'
    SUNKEN = 'sunken'
    SPHERICAL = 'spherical'
    OTHERS = 'others'


class Mushroom(BaseModel):
    """
    Mushroom model for POST json packet
    https://docs.pydantic.dev/latest/concepts/fields/
    """
    cap_diameter: float = Field(ge=0.38, le=62.34)
    cap_shape: CapShape = Field(...)
    has_ring: bool = Field(...)
    stem_height: float = Field(ge=0, le=33.92)
    stem_width: float = Field(ge=0, le=103.91)


def format_and_convert_input(
        cap_diameter: float = Query(ge=0.38, le=62.34),
        cap_shape: CapShape = Query(...),
        has_ring: bool = Query(...),
        stem_height: float = Query(ge=0, le=33.92),
        stem_width: float = Query(ge=0, le=103.91)):
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

    return pd.DataFrame({
        "cap-diameter": [cap_diameter],
        "cap-shape": [cap_shape_map[cap_shape]],
        "has-ring": [has_ring],
        "stem-height": [stem_height],
        "stem-width": [stem_width]
    })


def get_prediction(df, model=MODEL, preprocessor=PREPROCESSOR):
    """
    Return a prediction from a a preformatted df Returns 1 if poisonous,
    0 if edible NOT FOR REAL MUSHROOMS!
    """
    pred = model.predict(preprocessor.transform(df))

    return {'prediction': int(pred[0])}


@app.get("/")
async def root():
    """return hello world"""
    return {"message": "Hello World"}


@app.get("/mushroom_query")
async def mushroom_query(
        cap_diameter: float = Query(ge=0.38, le=62.34),
        cap_shape: CapShape = Query(...),
        has_ring: bool = Query(...),
        stem_height: float = Query(ge=0, le=33.92),
        stem_width: float = Query(ge=0, le=103.91)):
    """Take in query parameters for mushroom model"""

    obs = format_and_convert_input(
        cap_diameter,
        cap_shape,
        has_ring,
        stem_height,
        stem_width)

    return get_prediction(obs)


@app.post("/mushroom_post")
async def mushroom_post(mushroom: Mushroom):
    """Take in Mushroom for mushroom model"""

    # remember that
    # object attributes use dot notation
    obs = format_and_convert_input(
        mushroom.cap_diameter,
        mushroom.cap_shape,
        mushroom.has_ring,
        mushroom.stem_height,
        mushroom.stem_width,
    )

    print(mushroom)
    print(obs)

    return get_prediction(obs)

if __name__ == "__main__":

    TO_DO = """
    * Final App with Validation
    """

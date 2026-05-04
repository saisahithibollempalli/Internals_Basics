import argparse
import joblib
import pandas as pd

model = joblib.load("../models/model.pkl")

parser = argparse.ArgumentParser()

parser.add_argument("--slope_degrees", type=float, required=True)
parser.add_argument("--rainfall_mm", type=float, required=True)
parser.add_argument("--soil_depth_m", type=float, required=True)
parser.add_argument("--vegetation_index", type=float, required=True)

args = parser.parse_args()

df = pd.DataFrame([{
    "slope_degrees": args.slope_degrees,
    "rainfall_mm": args.rainfall_mm,
    "soil_depth_m": args.soil_depth_m,
    "vegetation_index": args.vegetation_index
}])

pred = model.predict(df)[0]

print(pred)
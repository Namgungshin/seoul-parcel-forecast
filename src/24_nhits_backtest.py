import argparse
import logging
import time
import warnings

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=int, required=True)  # 입력 길이(일)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--start", required=True)            # 첫 예측 시점(일요일)
parser.add_argument("--end", required=True)              # 마지막 예측 시점(일요일)
args = parser.parse_args()

panel = pd.read_parquet("reports/dl_panel.parquet")
EXOG = [c for c in panel.columns if c not in ("unique_id", "ds", "y", "available_mask")]
origins = pd.date_range(args.start, args.end, freq="W-SUN")
assert origins[-1] <= pd.Timestamp("2022-12-18"), "2023년을 건드리는 시점입니다"

rows, began = [], time.time()
for i, origin in enumerate(origins, 1):
    train = panel[panel["ds"] <= origin]
    future = panel[(panel["ds"] > origin) & (panel["ds"] <= origin + pd.Timedelta(days=7))]
    model = NHITS(
        h=7, input_size=args.input, futr_exog_list=EXOG,
        loss=MQLoss(quantiles=[0.5, 0.8, 0.84]), scaler_type="robust",
        max_steps=1000, random_seed=args.seed, accelerator="cpu",
        enable_progress_bar=False, logger=False, enable_checkpointing=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train)
    pred = nf.predict(df=train, futr_df=future[["unique_id", "ds"] + EXOG])
    pred = pred[pred["unique_id"].isin(["total", "fashion"])]
    pred = pred.rename(columns={"unique_id": "series", "ds": "target", "NHITS-median": "p50",
                                "NHITS-hi-60.0": "q80", "NHITS-hi-68.0": "q84"})
    rows.append(pred.assign(origin=origin))
    print(f"{i}/{len(origins)} {origin.date()} {time.time() - began:.0f}초", flush=True)

out = f"reports/nhits_i{args.input}_s{args.seed}_{origins[0].year}.csv"
pd.concat(rows, ignore_index=True).to_csv(out, index=False)
print("저장:", out)
from noshow_iq.preprocess import load_and_clean, get_features_and_target
from noshow_iq.model import train

df = load_and_clean("data/KaggleV2-May-2016.csv")
X, y = get_features_and_target(df)
metrics = train(X, y)
print("Training complete!")
print(metrics)
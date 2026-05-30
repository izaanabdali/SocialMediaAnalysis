```python
# ============================================================
# Linear Regression: Predict Engagement Rate (%)
# Platforms: Instagram, TikTok, Facebook
#
# DATASETS USED (raw tables, not pivot tables):
#   1. cleaned_social_engagement.xlsx → sheet "in"
#      3,025 post-level records | platform, content_type,
#      category, engagement_rate, follower_count, likes, etc.
#      → Used for: content_type + platform → engagement rate
#
#   2. Cleaned Engagement.xlsx → sheet "Sheet1"
#      1,000 user-level records | age, platform,
#      video_category, engagement_rate + behavioral features
#      → Used for: age_group + content_type + platform → ER
#
# Both models use log-transformed target (log1p) because
# engagement_rate is heavily right-skewed.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── LOAD RAW DATA TABLES ─────────────────────────────────────
# Table 1: Post-level social engagement (3,025 rows)
df_social = pd.read_excel("cleaned_social_engagement.xlsx", sheet_name="in")
df_social["platform"] = df_social["platform"].str.lower()

# Table 2: User-level engagement with age (1,000 rows)
df_user = pd.read_excel("Cleaned Engagement.xlsx", sheet_name="Sheet1")
df_user = df_user[df_user["Platform"].isin(["Instagram", "TikTok", "Facebook"])].copy()

print("Table 1 shape (social engagement): ", df_social.shape)
print("Table 2 shape (user engagement):   ", df_user.shape)

# Helper to encode labels safely
def safe_encode(le, vals):
    classes = list(le.classes_)
    return [classes.index(v) if v in classes else 0 for v in vals]


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODEL 1 — Content Type + Platform → Engagement Rate        ║
# ║  Dataset: cleaned_social_engagement.xlsx → "in" sheet        ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("MODEL 1: Content Type + Platform → Engagement Rate")
print("Dataset: cleaned_social_engagement.xlsx → 'in' sheet")
print("="*60)

df1 = df_social.copy()

# Log-transform target (heavily right-skewed: median=3%, max=261%)
df1["log_er"] = np.log1p(df1["engagement_rate"])

# Encode categoricals
le1_plat = LabelEncoder()
le1_ct   = LabelEncoder()
le1_cat  = LabelEncoder()
le1_tier = LabelEncoder()

df1["plat_enc"] = le1_plat.fit_transform(df1["platform"])
df1["ct_enc"]   = le1_ct.fit_transform(df1["content_type"])
df1["cat_enc"]  = le1_cat.fit_transform(df1["category"])
df1["tier_enc"] = le1_tier.fit_transform(df1["influencer_tier"])

# Features: core question vars + behavioral/post-level numerics
feature_cols_1 = [
    "plat_enc",        # platform (instagram/tiktok/facebook)
    "ct_enc",          # content_type (video/story/carousel/image/text)
    "cat_enc",         # category (Gaming, Food, Travel, etc.)
    "follower_count",  # audience size (negatively correlated with ER)
    "likes",
    "comments",
    "shares",
    "views",
    "saves",
    "hashtag_count",
    "content_length",
    "tier_enc",        # influencer tier (Nano/Micro/Mid-tier/Macro)
    "hour_of_day",
]

X1 = df1[feature_cols_1]
y1 = df1["log_er"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)

y1_pred_log  = model1.predict(X1_test)
y1_pred_orig = np.expm1(y1_pred_log)
y1_test_orig = np.expm1(y1_test)

r2_1_log  = r2_score(y1_test, y1_pred_log)
r2_1_orig = r2_score(y1_test_orig, y1_pred_orig)
mae_1     = mean_absolute_error(y1_test_orig, y1_pred_orig)
rmse_1    = np.sqrt(mean_squared_error(y1_test_orig, y1_pred_orig))

print(f"\n  R²  (log scale)     : {r2_1_log:.4f}")
print(f"  R²  (original %)    : {r2_1_orig:.4f}")
print(f"  MAE (original %)    : {mae_1:.4f}")
print(f"  RMSE(original %)    : {rmse_1:.4f}")

# Coefficients
coef1 = pd.DataFrame({
    "Feature": feature_cols_1,
    "Coefficient": model1.coef_
}).sort_values("Coefficient", ascending=False)
print("\n  Feature Coefficients:")
print(coef1.to_string(index=False))

# ── Segment Predictions: Content Type × Platform (medians for numerics) ──
platforms_1     = ["instagram", "tiktok", "facebook"]
content_types_1 = df1["content_type"].unique()
categories_1    = df1["category"].unique()

med1 = df1[["follower_count","likes","comments","shares","views",
            "saves","hashtag_count","content_length","hour_of_day"]].median()

rows1 = []
for pl in platforms_1:
    for ct in content_types_1:
        for cat in categories_1:
            rows1.append({"platform": pl, "content_type": ct, "category": cat})

grid1 = pd.DataFrame(rows1)
for col in ["follower_count","likes","comments","shares","views",
            "saves","hashtag_count","content_length","hour_of_day"]:
    grid1[col] = med1[col]
grid1["tier_enc"] = le1_tier.transform(["Mid-tier"] * len(grid1))

grid1["plat_enc"] = safe_encode(le1_plat, grid1["platform"])
grid1["ct_enc"]   = safe_encode(le1_ct,   grid1["content_type"])
grid1["cat_enc"]  = safe_encode(le1_cat,  grid1["category"])

grid1["Predicted_ER_%"] = np.expm1(model1.predict(grid1[feature_cols_1])).round(4)

# Best platform per content_type × category
best1 = (
    grid1.loc[grid1.groupby(["content_type","category"])["Predicted_ER_%"].idxmax()]
    [["content_type","category","platform","Predicted_ER_%"]]
    .rename(columns={"platform":"Best Platform","Predicted_ER_%":"Predicted ER %"})
    .sort_values(["content_type","category"])
    .reset_index(drop=True)
)

print("\n  Best Platform per Content Type × Category:")
print(best1.to_string(index=False))

print("\n  Platform Win Counts (Model 1):")
print(best1["Best Platform"].value_counts().to_string())


# ╔══════════════════════════════════════════════════════════════╗
# ║  MODEL 2 — Age Group + Content Type + Platform → ER         ║
# ║  Dataset: Cleaned Engagement.xlsx → "Sheet1"                 ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "="*60)
print("MODEL 2: Age Group + Content Type + Platform → Engagement Rate")
print("Dataset: Cleaned Engagement.xlsx → 'Sheet1'")
print("="*60)

df2 = df_user.copy()

# Bin age into standard groups
bins_ag   = [17, 24, 34, 44, 54, 65]
labels_ag = ["18-24", "25-34", "35-44", "45-54", "55-64"]
df2["Age Group"] = pd.cut(df2["Age"], bins=bins_ag, labels=labels_ag)

# Log-transform target
df2["log_er"] = np.log1p(df2["Engagement Rate"])

# Encode categoricals
le2_plat = LabelEncoder()
le2_vcat = LabelEncoder()
le2_ag   = LabelEncoder()

df2["plat_enc"] = le2_plat.fit_transform(df2["Platform"])
df2["vcat_enc"] = le2_vcat.fit_transform(df2["Video Category"])
df2["ag_enc"]   = le2_ag.fit_transform(df2["Age Group"].astype(str))

# Features: core question vars + behavioral numerics
feature_cols_2 = [
    "ag_enc",                    # age group (binned)
    "vcat_enc",                  # video category / content type
    "plat_enc",                  # platform
    "Age",                       # continuous age
    "Video Length",
    "Total Time Spent",
    "Number of Videos Watched",
    "Engagement",                # raw engagement count (corr=0.69 with log ER)
]

X2 = df2[feature_cols_2]
y2 = df2["log_er"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)

y2_pred_log  = model2.predict(X2_test)
y2_pred_orig = np.expm1(y2_pred_log)
y2_test_orig = np.expm1(y2_test)

r2_2_log  = r2_score(y2_test, y2_pred_log)
r2_2_orig = r2_score(y2_test_orig, y2_pred_orig)
mae_2     = mean_absolute_error(y2_test_orig, y2_pred_orig)
rmse_2    = np.sqrt(mean_squared_error(y2_test_orig, y2_pred_orig))

print(f"\n  R²  (log scale)     : {r2_2_log:.4f}")
print(f"  R²  (original %)    : {r2_2_orig:.4f}")
print(f"  MAE (original %)    : {mae_2:.2f}")
print(f"  RMSE(original %)    : {rmse_2:.2f}")

# Coefficients
coef2 = pd.DataFrame({
    "Feature": feature_cols_2,
    "Coefficient": model2.coef_
}).sort_values("Coefficient", ascending=False)
print("\n  Feature Coefficients:")
print(coef2.to_string(index=False))

# ── Segment Predictions: Age Group × Video Category × Platform ─
age_groups_2  = labels_ag
video_cats_2  = df2["Video Category"].unique()
platforms_2   = ["Instagram", "TikTok", "Facebook"]

med2 = df2[["Video Length","Total Time Spent","Number of Videos Watched","Engagement"]].median()
age_median_map = df2.groupby("Age Group", observed=True)["Age"].median()

rows2 = []
for ag in age_groups_2:
    for vc in video_cats_2:
        for pl in platforms_2:
            rows2.append({"Age Group": ag, "Video Category": vc, "Platform": pl})

grid2 = pd.DataFrame(rows2)
grid2["Age"] = grid2["Age Group"].map(age_median_map)
for col in ["Video Length","Total Time Spent","Number of Videos Watched","Engagement"]:
    grid2[col] = med2[col]

grid2["ag_enc"]   = safe_encode(le2_ag,   grid2["Age Group"].astype(str))
grid2["vcat_enc"] = safe_encode(le2_vcat, grid2["Video Category"])
grid2["plat_enc"] = safe_encode(le2_plat, grid2["Platform"])

grid2["Predicted_ER_%"] = np.expm1(model2.predict(grid2[feature_cols_2])).round(2)

# Best platform per Age Group × Video Category
best2 = (
    grid2.loc[grid2.groupby(["Age Group","Video Category"])["Predicted_ER_%"].idxmax()]
    [["Age Group","Video Category","Platform","Predicted_ER_%"]]
    .rename(columns={"Platform":"Best Platform","Predicted_ER_%":"Predicted ER %"})
    .sort_values(["Age Group","Video Category"])
    .reset_index(drop=True)
)

print("\n  Best Platform per Age Group × Video Category:")
print(best2.to_string(index=False))

print("\n  Platform Win Counts (Model 2):")
print(best2["Best Platform"].value_counts().to_string())

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
m1_winner = best1["Best Platform"].value_counts().idxmax()
m2_winner = best2["Best Platform"].value_counts().idxmax()
print(f"""
  MODEL 1 (Content Type + Platform → ER)
    Source  : cleaned_social_engagement.xlsx → 'in' (3,025 rows)
    R² log  : {r2_1_log:.4f}   |   R² original: {r2_1_orig:.4f}

  MODEL 2 (Age Group + Content Type + Platform → ER)
    Source  : Cleaned Engagement.xlsx → 'Sheet1' (750 rows after filter)
    R² log  : {r2_2_log:.4f}   |   R² original: {r2_2_orig:.4f}

  ANSWER:
    Yes — engagement rate is predictable with strong accuracy
    (R² 0.83–0.88 on log scale) from platform, content type,
    and age group combined with behavioral signals.

    Highest predicted engagement platform across all segments:
      Model 1 (content type focus) → {m1_winner}
      Model 2 (age group focus)    → {m2_winner}
""")

# Save CSV outputs
grid1[["platform","content_type","category","Predicted_ER_%"]].sort_values(
    ["platform","content_type","category"]
).to_csv("model1_all_predictions.csv", index=False)
best1.to_csv("model1_best_platform_per_segment.csv", index=False)

grid2[["Age Group","Video Category","Platform","Predicted_ER_%"]].sort_values(
    ["Age Group","Video Category","Platform"]
).to_csv("model2_all_predictions.csv", index=False)
best2.to_csv("model2_best_platform_per_segment.csv", index=False)

print("  Outputs saved to data/")
```

    Table 1 shape (social engagement):  (3025, 18)
    Table 2 shape (user engagement):    (750, 14)
    
    ============================================================
    MODEL 1: Content Type + Platform → Engagement Rate
    Dataset: cleaned_social_engagement.xlsx → 'in' sheet
    ============================================================
    
      R²  (log scale)     : 0.8315
      R²  (original %)    : 0.5353
      MAE (original %)    : 7.0847
      RMSE(original %)    : 25.2432
    
      Feature Coefficients:
           Feature   Coefficient
          tier_enc  4.320844e-01
          plat_enc  6.523681e-02
            ct_enc  7.220950e-03
           cat_enc  3.173886e-03
     hashtag_count  2.902553e-04
          comments  7.375844e-05
            shares  6.695082e-05
             likes  4.776932e-05
             saves  2.242627e-05
    content_length  1.410828e-05
             views  1.350630e-07
    follower_count -3.259293e-06
       hour_of_day -8.853931e-05
    
      Best Platform per Content Type × Category:
    content_type      category Best Platform  Predicted ER %
        carousel      Business        tiktok          7.7717
        carousel     Education        tiktok          7.7996
        carousel Entertainment        tiktok          7.8276
        carousel       Fashion        tiktok          7.8556
        carousel       Fitness        tiktok          7.8838
        carousel          Food        tiktok          7.9120
        carousel        Gaming        tiktok          7.9404
        carousel        Health        tiktok          7.9688
        carousel     Lifestyle        tiktok          7.9973
        carousel        Sports        tiktok          8.0259
        carousel    Technology        tiktok          8.0546
        carousel        Travel        tiktok          8.0834
           image      Business        tiktok          7.8353
           image     Education        tiktok          7.8634
           image Entertainment        tiktok          7.8915
           image       Fashion        tiktok          7.9198
           image       Fitness        tiktok          7.9482
           image          Food        tiktok          7.9766
           image        Gaming        tiktok          8.0051
           image        Health        tiktok          8.0338
           image     Lifestyle        tiktok          8.0625
           image        Sports        tiktok          8.0913
           image    Technology        tiktok          8.1202
           image        Travel        tiktok          8.1492
           story      Business        tiktok          7.8993
           story     Education        tiktok          7.9276
           story Entertainment        tiktok          7.9560
           story       Fashion        tiktok          7.9845
           story       Fitness        tiktok          8.0130
           story          Food        tiktok          8.0417
           story        Gaming        tiktok          8.0704
           story        Health        tiktok          8.0992
           story     Lifestyle        tiktok          8.1282
           story        Sports        tiktok          8.1572
           story    Technology        tiktok          8.1863
           story        Travel        tiktok          8.2155
            text      Business        tiktok          7.9638
            text     Education        tiktok          7.9923
            text Entertainment        tiktok          8.0209
            text       Fashion        tiktok          8.0496
            text       Fitness        tiktok          8.0783
            text          Food        tiktok          8.1072
            text        Gaming        tiktok          8.1361
            text        Health        tiktok          8.1652
            text     Lifestyle        tiktok          8.1943
            text        Sports        tiktok          8.2235
            text    Technology        tiktok          8.2529
            text        Travel        tiktok          8.2823
           video      Business        tiktok          8.0288
           video     Education        tiktok          8.0575
           video Entertainment        tiktok          8.0863
           video       Fashion        tiktok          8.1151
           video       Fitness        tiktok          8.1441
           video          Food        tiktok          8.1732
           video        Gaming        tiktok          8.2024
           video        Health        tiktok          8.2316
           video     Lifestyle        tiktok          8.2610
           video        Sports        tiktok          8.2904
           video    Technology        tiktok          8.3199
           video        Travel        tiktok          8.3496
    
      Platform Win Counts (Model 1):
    Best Platform
    tiktok    60
    
    ============================================================
    MODEL 2: Age Group + Content Type + Platform → Engagement Rate
    Dataset: Cleaned Engagement.xlsx → 'Sheet1'
    ============================================================
    
      R²  (log scale)     : 0.8806
      R²  (original %)    : 0.7221
      MAE (original %)    : 23.47
      RMSE(original %)    : 49.99
    
      Feature Coefficients:
                     Feature  Coefficient
                      ag_enc     0.039406
                    plat_enc     0.024493
                Video Length     0.006270
    Number of Videos Watched     0.000940
                  Engagement     0.000278
                    vcat_enc    -0.001685
                         Age    -0.004599
            Total Time Spent    -0.008508
    
      Best Platform per Age Group × Video Category:
    Age Group Video Category Best Platform  Predicted ER %
        18-24           ASMR        TikTok           35.38
        18-24         Comedy        TikTok           35.32
        18-24  Entertainment        TikTok           35.25
        18-24         Gaming        TikTok           35.19
        18-24    Jokes/Memes        TikTok           35.13
        18-24     Life Hacks        TikTok           35.07
        18-24         Pranks        TikTok           35.01
        18-24         Trends        TikTok           34.95
        18-24          Vlogs        TikTok           34.89
        25-34           ASMR        TikTok           35.47
        25-34         Comedy        TikTok           35.41
        25-34  Entertainment        TikTok           35.35
        25-34         Gaming        TikTok           35.29
        25-34    Jokes/Memes        TikTok           35.23
        25-34     Life Hacks        TikTok           35.17
        25-34         Pranks        TikTok           35.11
        25-34         Trends        TikTok           35.04
        25-34          Vlogs        TikTok           34.98
        35-44           ASMR        TikTok           35.07
        35-44         Comedy        TikTok           35.01
        35-44  Entertainment        TikTok           34.95
        35-44         Gaming        TikTok           34.88
        35-44    Jokes/Memes        TikTok           34.82
        35-44     Life Hacks        TikTok           34.76
        35-44         Pranks        TikTok           34.70
        35-44         Trends        TikTok           34.64
        35-44          Vlogs        TikTok           34.58
        45-54           ASMR        TikTok           34.83
        45-54         Comedy        TikTok           34.77
        45-54  Entertainment        TikTok           34.71
        45-54         Gaming        TikTok           34.65
        45-54    Jokes/Memes        TikTok           34.59
        45-54     Life Hacks        TikTok           34.53
        45-54         Pranks        TikTok           34.47
        45-54         Trends        TikTok           34.41
        45-54          Vlogs        TikTok           34.35
        55-64           ASMR        TikTok           34.60
        55-64         Comedy        TikTok           34.54
        55-64  Entertainment        TikTok           34.48
        55-64         Gaming        TikTok           34.42
        55-64    Jokes/Memes        TikTok           34.36
        55-64     Life Hacks        TikTok           34.30
        55-64         Pranks        TikTok           34.24
        55-64         Trends        TikTok           34.18
        55-64          Vlogs        TikTok           34.12
    
      Platform Win Counts (Model 2):
    Best Platform
    TikTok    45
    
    ============================================================
    FINAL SUMMARY
    ============================================================
    
      MODEL 1 (Content Type + Platform → ER)
        Source  : cleaned_social_engagement.xlsx → 'in' (3,025 rows)
        R² log  : 0.8315   |   R² original: 0.5353
    
      MODEL 2 (Age Group + Content Type + Platform → ER)
        Source  : Cleaned Engagement.xlsx → 'Sheet1' (750 rows after filter)
        R² log  : 0.8806   |   R² original: 0.7221
    
      ANSWER:
        Yes — engagement rate is predictable with strong accuracy
        (R² 0.83–0.88 on log scale) from platform, content type,
        and age group combined with behavioral signals.
    
        Highest predicted engagement platform across all segments:
          Model 1 (content type focus) → tiktok
          Model 2 (age group focus)    → TikTok
    
      Outputs saved to data/



```python

```

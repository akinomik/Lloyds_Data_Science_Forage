from catboost import CatBoostClassifier, Pool
from functools import reduce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


# pipenv install xyz - to ensure via pipfile

# load all 4 sheets from Excel file + ensure customerID is string
path = 'cust.xlsx'
sheets = pd.read_excel(path, sheet_name=None, dtype={'CustomerID': str})

demo   = sheets["Customer_Demographics"].copy()
tx     = sheets["Transaction_History"].copy()
svc    = sheets["Customer_Service"].copy()
online = sheets["Online_Activity"].copy()
churn  = sheets["Churn_Status"].copy()

# ---- Parse dates
tx["TransactionDate"] = pd.to_datetime(tx["TransactionDate"], errors="coerce")
svc["InteractionDate"] = pd.to_datetime(svc["InteractionDate"], errors="coerce")
online["LastLoginDate"] = pd.to_datetime(online["LastLoginDate"], errors="coerce")

# ---- Numeric coercions
tx["AmountSpent"] = pd.to_numeric(tx["AmountSpent"], errors="coerce")
online["LoginFrequency"] = pd.to_numeric(online["LoginFrequency"], errors="coerce")
# IMPORTANT FIX: keep ServiceUsage as CATEGORICAL; DO NOT coerce to numeric
# online["ServiceUsage"]  <-- leave as text (Mobile App / Website / Online Banking)

# ---- Snapshot for recency (set explicitly if you have a cutoff date)
snapshot_date = pd.Series([
    tx["TransactionDate"].max(),
    svc["InteractionDate"].max(),
    online["LastLoginDate"].max()
]).max()

# ===============================================
# B) CUSTOMER-LEVEL AGGREGATION (for modelling)
# ===============================================

# Transactions -> per-customer features
if not tx.empty:
    tx_sorted = tx.sort_values(["CustomerID","TransactionDate"])
    tx_gap = (tx_sorted.groupby("CustomerID")["TransactionDate"]
                        .apply(lambda s: s.diff().dt.days.dropna().mean())
                        .rename("tx_avg_gap_days").reset_index())
    tx_agg = (tx.groupby("CustomerID").agg(
                txn_count=("TransactionID","count"),
                amount_sum=("AmountSpent","sum"),
                amount_mean=("AmountSpent","mean"),
                amount_std=("AmountSpent","std"),
                first_txn_dt=("TransactionDate","min"),
                last_txn_dt=("TransactionDate","max"),
                product_cat_nunique=("ProductCategory","nunique")
             ).reset_index())
    tx_agg["tx_recency_days"] = (snapshot_date - tx_agg["last_txn_dt"]).dt.days
    tx["ym"] = tx["TransactionDate"].dt.to_period("M")
    active_months = (tx.dropna(subset=["ym"])
                       .groupby("CustomerID")["ym"].nunique()
                       .rename("active_months").reset_index())
    tx_agg = tx_agg.merge(tx_gap, on="CustomerID", how="left").merge(active_months, on="CustomerID", how="left")
else:
    tx_agg = pd.DataFrame(columns=["CustomerID"])

# Customer Service -> per-customer features
# this code counts complaints and unresolved issues
# also computes recency of last interaction and types of interactions
if not svc.empty:
    svc_agg = (svc.groupby("CustomerID").agg(
                cs_count=("InteractionID","count"),
                cs_types_nunique=("InteractionType","nunique"),
                last_cs_dt=("InteractionDate","max"),
                unresolved_count=("ResolutionStatus", lambda s: (s.str.lower()!="resolved").sum())
             ).reset_index())
    svc_agg["cs_recency_days"] = (snapshot_date - svc_agg["last_cs_dt"]).dt.days
    comp = (svc.assign(is_complaint=svc["InteractionType"].str.lower().eq("complaint"))
              .groupby("CustomerID")["is_complaint"].sum()
              .rename("complaint_count").reset_index())
    svc_agg = svc_agg.merge(comp, on="CustomerID", how="left")
else:
    svc_agg = pd.DataFrame(columns=["CustomerID"])

# Online (1:1) -> add recency BUT KEEP ServiceUsage as text
# this code computes recency of last login only
online_agg = online.copy()
if not online_agg.empty and "LastLoginDate" in online_agg:
    online_agg["lastlogin_recency_days"] = (snapshot_date - online_agg["LastLoginDate"]).dt.days

# ---- Final customer table (one row per customer)
# merge all together on CustomerID and validate 1:1 relationships
final = (demo
         .merge(churn, on="CustomerID", how="left", validate="one_to_one")
         .merge(online_agg, on="CustomerID", how="left", validate="one_to_one"))

# merge in tx_agg and svc_agg if they are not empty
if not tx_agg.empty:
    final = final.merge(tx_agg, on="CustomerID", how="left", validate="one_to_one")
if not svc_agg.empty:
    final = final.merge(svc_agg, on="CustomerID", how="left", validate="one_to_one")

# ---- Integrity & conservation checks
# e.g. no duplicate customers, sums of counts match original tables
assert final.duplicated("CustomerID").sum() == 0
if "txn_count" in final: print("TX rows:", len(tx), "| Sum txn_count in final:", final["txn_count"].sum())
if "cs_count"  in final: print("SVC rows:", len(svc), "| Sum cs_count in final:", final["cs_count"].sum())
print("Final shape:", final.shape)

# Exactly one row per customer
assert final["CustomerID"].nunique() == len(final) == 1000

# Fan-out check
vc = final["CustomerID"].value_counts()
print(vc.mean(), vc.max(), '\n')  # should be 1.0 and 1

# Conservation
assert final["txn_count"].sum() == len(tx)
assert final["cs_count"].sum()  == len(svc)

pd.set_option("display.max_columns", 200)

# ===================================================
# C) EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================

# raw tables - quality checks
for name, df in {"demo":demo, "tx":tx, "svc":svc, "online":online, "churn":churn}.items():
    print(f"\n[{name}] shape={df.shape}")
    print(df.isna().mean().sort_values(ascending=False).head(8))

print("\nUnique customers per table:",
      {k: v["CustomerID"].nunique() for k,v in {"demo":demo,"tx":tx,"svc":svc,"online":online,"churn":churn}.items()})
print("\nDuplicate TransactionID:", tx["TransactionID"].duplicated().sum())
print("Duplicate InteractionID:", svc["InteractionID"].duplicated().sum())
print("\nNegative or zero amounts:", (tx["AmountSpent"]<=0).sum())
print("Future TX dates:", (tx["TransactionDate"] > pd.Timestamp.today()).sum())
print("Future SVC dates:", (svc["InteractionDate"] > pd.Timestamp.today()).sum())

for col in ["Gender","MaritalStatus","IncomeLevel"]:
    if col in demo:
        print(col, demo[col].value_counts(dropna=False))
if "ResolutionStatus" in svc:
    print("ResolutionStatus:", svc["ResolutionStatus"].value_counts(dropna=False))
if "InteractionType" in svc:
    print("InteractionType:", svc["InteractionType"].value_counts(dropna=False))

# DEDUP sanity (we already saw they are fine, keep for completeness)
tx_before, svc_before = len(tx), len(svc)
tx = tx.drop_duplicates()
svc = svc.drop_duplicates()
print(f"TX dropped exact dup rows: {tx_before - len(tx)}")
print(f"SVC dropped exact dup rows: {svc_before - len(svc)}")

# Within-customer event numbering
tx = tx.sort_values(["CustomerID","TransactionDate"]).copy()
tx["tx_event_no"] = tx.groupby("CustomerID").cumcount() + 1
svc = svc.sort_values(["CustomerID","InteractionDate"]).copy()
svc["svc_event_no"] = svc.groupby("CustomerID").cumcount() + 1

print("TX rows now:", len(tx))
print("SVC rows now:", len(svc))
print("Within-customer duplicated (CustomerID, TransactionID) in tx:",
      tx.duplicated(subset=["CustomerID","TransactionID"]).sum())
print("Within-customer duplicated (CustomerID, InteractionID) in svc:",
      svc.duplicated(subset=["CustomerID","InteractionID"]).sum())
print('\n')

# Keep all labelled customers (1000 here but can be any number)
base = demo.merge(churn, on="CustomerID", how="inner", validate="one_to_one")
final = (base
         .merge(online_agg, on="CustomerID", how="left", validate="one_to_one")
         .merge(tx_agg,     on="CustomerID", how="left", validate="one_to_one")
         .merge(svc_agg,    on="CustomerID", how="left", validate="one_to_one"))
print("Final shape:", final.shape)
assert final["CustomerID"].nunique() == len(final) == 1000, "Expected 1 row per customer"

# Coverage flags - whether customer had any transactions or service interactions
final["had_tx"] = final["txn_count"].notna().astype(int)
final["had_service"] = final["cs_count"].notna().astype(int)

# Minimal fills for EDA convenience
for c in ["txn_count","product_cat_nunique","cs_count","cs_types_nunique","complaint_count","unresolved_count","active_months"]:
    if c in final: final[c] = final[c].fillna(0)

for c in ["tx_recency_days","cs_recency_days","lastlogin_recency_days"]:
    if c in final:
        mx = final[c].max(skipna=True)
        final[c] = final[c].fillna(mx + 1)

print(final[["had_tx","had_service"]].value_counts(dropna=False))
print("Churn rate:", final["ChurnStatus"].mean().round(3))
print("Sum txn_count:", final.get("txn_count", pd.Series(dtype=float)).sum())
print("Sum cs_count:", final.get("cs_count", pd.Series(dtype=float)).sum())

# ===============================
# EVENT-LEVEL PLOTS
# ===============================

# Transaction amounts, weekly counts, inter-transaction gaps - temp copies to avoid modifying originals
_tx = tx.copy()
_svc = svc.copy()
_tx["TransactionDate"] = pd.to_datetime(_tx["TransactionDate"], errors="coerce")
_svc["InteractionDate"] = pd.to_datetime(_svc["InteractionDate"], errors="coerce")

fig = plt.figure(figsize=(6,4))
_tx["AmountSpent"].dropna().hist(bins=50)
plt.title("AmountSpent"); plt.xlabel("Amount"); plt.ylabel("Frequency"); plt.tight_layout()

fig = plt.figure(figsize=(6,4))
np.log1p(_tx["AmountSpent"].clip(lower=0)).dropna().hist(bins=50)
plt.title("log1p(AmountSpent)"); plt.xlabel("log1p(amount)"); plt.ylabel("Frequency"); plt.tight_layout()

_tx_weekly = (_tx.dropna(subset=["TransactionDate"])
                .groupby(_tx["TransactionDate"].dt.to_period("W"))
                .size())
if not _tx_weekly.empty:
    x = _tx_weekly.index.to_timestamp()
    y = _tx_weekly.values
    y_ma = _tx_weekly.rolling(4).mean().values
    plt.figure(figsize=(7,3.5)); plt.plot(x, y); plt.plot(x, y_ma, linestyle="--")
    plt.legend(["Weekly count","4-week MA"], frameon=False)
    plt.title("Weekly Transactions"); plt.ylabel("Count"); plt.tight_layout()

_tx_sorted = _tx.sort_values(["CustomerID","TransactionDate"])
gaps = (_tx_sorted.groupby("CustomerID")["TransactionDate"]
                  .apply(lambda s: s.diff().dt.days.dropna()))
if not gaps.empty:
    fig = plt.figure(figsize=(6,4))
    gaps.hist(bins=50)
    plt.title("Inter-transaction gap (days)")
    plt.xlabel("Days between consecutive transactions")
    plt.ylabel("Customers x occurrences"); plt.tight_layout()

_svc_weekly = (_svc.dropna(subset=["InteractionDate"])
                 .groupby(_svc["InteractionDate"].dt.to_period("W"))
                 .size())
if not _svc_weekly.empty:
    x = _svc_weekly.index.to_timestamp()
    y = _svc_weekly.values
    y_ma = _svc_weekly.rolling(4).mean().values
    plt.figure(figsize=(7,3.5)); plt.plot(x, y); plt.plot(x, y_ma, linestyle="--")
    plt.legend(["Weekly service","4-week MA"], frameon=False)
    plt.title("Weekly Service Interactions"); plt.ylabel("Count"); plt.tight_layout()

# ===============================
# CUSTOMER-LEVEL PLOTS
# ===============================
def safe_boxplot(df, x, y, title=None):
    '''
    Safe boxplot: skips if any group has no valid numeric data
    '''
    tmp = df[[x, y]].copy()
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")  # numeric only
    counts = tmp.groupby(x)[y].apply(lambda s: s.notna().sum())
    missing_groups = counts[counts == 0].index.tolist()
    if len(counts) < tmp[x].nunique() or len(missing_groups) > 0:
        print(f"Skipping '{y}': empty/invalid data for groups {missing_groups}")
        return
    ax = sns.boxplot(data=tmp, x=x, y=y)
    ax.set_title(title or f"{y} by {x}")
    plt.show()

# Numeric candidates (EXCLUDE ServiceUsage here; it's categorical)
cands = [c for c in ["txn_count","amount_sum","amount_mean","tx_recency_days",
                     "active_months","product_cat_nunique",
                     "cs_count","complaint_count","unresolved_count","cs_recency_days",
                     "LoginFrequency","lastlogin_recency_days"]
         if c in final.columns]

for c in cands:
    safe_boxplot(final, "ChurnStatus", c, title=f"{c} by ChurnStatus")

# Categorical EDA for ServiceUsage
if "ServiceUsage" in final.columns:
    ct = pd.crosstab(final["ServiceUsage"], final["ChurnStatus"], normalize="index")
    print("\nChurn share by ServiceUsage channel:\n", ct)
    ax = ct.plot(kind="bar", stacked=True, title="ServiceUsage vs Churn", figsize=(6,4))
    ax.set_ylabel("Share"); plt.tight_layout(); plt.show()

# ================================================
# ONE-HOT ENCODING for modelling later
# ================================================
# Convert ServiceUsage into dummies (keep NA as a category flag)
if "ServiceUsage" in final.columns:
    service_dummies = pd.get_dummies(final["ServiceUsage"], prefix="channel", dummy_na=True)
    final = pd.concat([final.drop(columns=["ServiceUsage"]), service_dummies], axis=1)

# ===============================================
# D) PREPARATION FOR MODELLING
# ===============================================
# ---- cleaning / imputation / outliers

# Missingness flags (can carry signal)
num_cols = final.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    final[f"{c}_was_missing"] = final[c].isna().astype(int)

# Impute counts/diversities to 0 (already mostly done)
for c in ["txn_count","product_cat_nunique","cs_count","cs_types_nunique",
          "complaint_count","unresolved_count","active_months"]:
    if c in final: final[c] = final[c].fillna(0)

# Recencies: fill “no history” with a large constant (max+1 already applied earlier)
# Monetary stats: winsorize light to reduce tail influence
for c in ["amount_sum","amount_mean","amount_std"]:
    if c in final:
        q99 = final[c].quantile(0.99)
        final[c] = final[c].clip(upper=q99)

# log transform amounts for linear models
for c in ["amount_sum","amount_mean"]:
    if c in final:
        final[f"log1p_{c}"] = np.log1p(final[c])

        # ---- Ensure ServiceUsage exists before OHE (and reattach it if needed)
if "ServiceUsage" not in final.columns:
    # try to pull it from the Online_Activity sheet you already loaded
    if "ServiceUsage" in online.columns:
        final = final.merge(
            online[["CustomerID", "ServiceUsage"]],
            on="CustomerID",
            how="left",
            validate="one_to_one"
        )
    else:
        print("Warning: 'ServiceUsage' not found in final or online; skipping channel OHE.")

# ---- Only OHE if the column is present and not already encoded
if "ServiceUsage" in final.columns:
    # avoid double-encoding if channel_* already exists
    already_ohe = any(col.startswith("channel_") for col in final.columns)
    if not already_ohe:
        service_dummies = pd.get_dummies(final["ServiceUsage"], prefix="channel", dummy_na=True)
        final = pd.concat([final.drop(columns=["ServiceUsage"]), service_dummies], axis=1)
    else:
        print("Channel dummies already present; skipping OHE.")


# ---- encoding / scaling
# One-hot encode ServiceUsage (categorical channel)
service_dummies = pd.get_dummies(final["ServiceUsage"], prefix="channel", dummy_na=True)
final = pd.concat([final.drop(columns=["ServiceUsage"]), service_dummies], axis=1)


# ---- Fix the two NaN culprits + add flags
final["tx_avg_gap_days_missing"] = final["tx_avg_gap_days"].isna().astype(int)
final["amount_std_missing"]      = final["amount_std"].isna().astype(int)

# amount_std: when only 1 txn there is no variance; 0 is a natural fill
if "amount_std" in final:
    final["amount_std"] = final["amount_std"].fillna(0)

# tx_avg_gap_days: undefined when <2 tx; use median of defined gaps
if "tx_avg_gap_days" in final:
    med_gap = final["tx_avg_gap_days"].median(skipna=True)
    final["tx_avg_gap_days"] = final["tx_avg_gap_days"].fillna(med_gap)

# (Optional) quick sanity — should now be all zeros for these
print("NaN share after targeted fills:")
print(final[["tx_avg_gap_days","amount_std"]].isna().mean())


# Split X/y; keep ID out
y = final["ChurnStatus"].astype(int)
drop_cols = ["CustomerID","ChurnStatus","first_txn_dt","last_txn_dt","last_cs_dt","LastLoginDate"]
X = final.drop(columns=[c for c in drop_cols if c in final.columns])

# print(X.isna().mean().sort_values(ascending=False).head(10))    # still some NAs

# Show any remaining NaNs (should be 0.0 everywhere)
print("Top remaining NaN shares in X:")
print(X.isna().mean().sort_values(ascending=False).head(10))

# If you plan a linear/SVM baseline, scale numerics. Trees don’t need it.
num_feats = X.select_dtypes(include=[np.number]).columns.tolist()
cat_feats = X.select_dtypes(exclude=[np.number]).columns.tolist()  # should be empty now

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
], remainder="drop")  # all features are numeric after OHE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Logistic Regression baseline as pipeline
clf = Pipeline([
    ("prep", pre),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])

clf.fit(X_train, y_train)

# Evaluate by ROC-AUC
p = clf.predict_proba(X_test)[:,1]
print("ROC-AUC for LogReg:", f"{roc_auc_score(y_test, p):.3f}")
print(classification_report(y_test, (p>=0.5).astype(int)))

# ===============================================
# Some FEATURE ENGINEERING IMPROVEMENTS
# ROC-AUC so far is average ~0.495
# ===============================================
# adding a 30/60/90-day activity features & flags

# cutoffs relative to your snapshot_date (already defined)
cut30 = snapshot_date - pd.Timedelta(days=30)
cut60 = snapshot_date - pd.Timedelta(days=60)
cut90 = snapshot_date - pd.Timedelta(days=90)

# ---- Transactions windows
tx_30 = tx[tx["TransactionDate"] >= cut30].groupby("CustomerID").size().rename("tx_30").reset_index()
tx_60 = tx[tx["TransactionDate"] >= cut60].groupby("CustomerID").size().rename("tx_60").reset_index()
tx_90 = tx[tx["TransactionDate"] >= cut90].groupby("CustomerID").size().rename("tx_90").reset_index()

amt_30 = (tx[tx["TransactionDate"] >= cut30]
          .groupby("CustomerID")["AmountSpent"].sum().rename("amt_30").reset_index())
amt_90 = (tx[tx["TransactionDate"] >= cut90]
          .groupby("CustomerID")["AmountSpent"].sum().rename("amt_90").reset_index())

# ---- Service windows (complaints/unresolved in last 90d)
svc_recent = svc[svc["InteractionDate"] >= cut90].copy()
svc_recent["is_complaint"] = svc_recent["InteractionType"].str.lower().eq("complaint")
svc_recent["is_unresolved"] = svc_recent["ResolutionStatus"].str.lower() != "resolved"
svc_90 = (svc_recent.groupby("CustomerID")
          .agg(complaints_90d=("is_complaint","sum"),
               unresolved_90d=("is_unresolved","sum"))
          .reset_index())

# ---- Merge into final
for add in [tx_30, tx_60, tx_90, amt_30, amt_90, svc_90]:
    final = final.merge(add, on="CustomerID", how="left")

# ---- Fill + flags
for c in ["tx_30","tx_60","tx_90","amt_30","amt_90","complaints_90d","unresolved_90d"]:
    final[c] = final[c].fillna(0)

final["no_tx_90d"]      = (final["tx_90"] == 0).astype(int)
final["no_login_90d"]   = (final["lastlogin_recency_days"] > 90).astype(int)
final["complained_90d"] = (final["complaints_90d"] > 0).astype(int)

# safe ratios (avoid div/0)
final["avg_spend_per_tx"] = final["amount_sum"] / final["txn_count"].replace(0, np.nan)
final["avg_spend_per_tx"] = final["avg_spend_per_tx"].fillna(0)

y = final["ChurnStatus"].astype(int)
drop_cols = ["CustomerID","ChurnStatus","first_txn_dt","last_txn_dt","last_cs_dt","LastLoginDate"]
X = final.drop(columns=[c for c in drop_cols if c in final.columns])

# ----------------------------------------------------------------
# df_model is the final feature set for modelling - but ensure it has OHE done for RF, otherwise categoricals remain
# 'df_model' is for RF; 'final' is for CatBoost & LightGBM
# ----------------------------------------------------------------
df_model = final.drop(columns=[c for c in drop_cols if c in final.columns]).copy()

# One-hot encode ALL non-numeric columns (incl. Gender/MaritalStatus/IncomeLevel/ServiceUsage)
non_num = df_model.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
if non_num:
    df_model = pd.get_dummies(df_model, columns=non_num, dummy_na=True)

print(df_model.head(5).T)
print("Final shape for modelling:", df_model.shape)


# several categorials have many levels, leading to many dummies
# this can lead to duplicate column

def dedupe_columns(df, collapse="max"):
    """
    Ensures all column names are unique.
    If duplicates exist, optionally collapse duplicate dummy columns
    by row-wise 'max' (works for 0/1 dummies); then drop the rest.
    """
    cols = df.columns
    dupes = cols[cols.duplicated()].unique()
    for name in dupes:
        same = [c for c in df.columns if c == name]
        if collapse == "max" and len(same) > 1:
            df[name] = df[same].max(axis=1)  # collapse duplicates (esp. for dummies)
        df.drop(columns=same[1:], inplace=True)
    # If names still collide due to whitespace/case, normalize and recheck
    df.columns = df.columns.str.strip()
    return df


# =============================
# RF
# =============================
# Build df_model fresh
drop_cols = ["CustomerID","ChurnStatus","first_txn_dt","last_txn_dt","last_cs_dt","LastLoginDate"]
df_model = final.drop(columns=[c for c in drop_cols if c in final.columns]).copy()

# One-hot ALL non-numerics exactly once
non_num = df_model.select_dtypes(exclude=[np.number, "bool", "boolean"]).columns.tolist()
if non_num:
    df_model = pd.get_dummies(df_model, columns=non_num, dummy_na=True)

# Deduplicate any columns that slipped in from earlier runs/merges
df_model = dedupe_columns(df_model)

# Final NA guard
df_model = df_model.fillna(0)

rf = RandomForestClassifier(
    n_estimators=1000, min_samples_leaf=3, class_weight="balanced",
    random_state=42, n_jobs=-1
)

# Add cross-validation scores
cv_scores = cross_val_score(rf, df_model, y, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

y = final["ChurnStatus"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(df_model, y, stratify=y, test_size=0.25, random_state=42)

rf.fit(X_train, y_train)
p = rf.predict_proba(X_test)[:,1]
print("ROC-AUC:", f"{roc_auc_score(y_test, p):.3f}")
print("PR-AUC :", f"{average_precision_score(y_test, p):.3f}")

# threshold tuning (F1-best)
prec, rec, thr = precision_recall_curve(y_test, p)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1[:-1])]
print("Best threshold for RF:", best_thr)
print(classification_report(y_test, (p>=best_thr).astype(int), digits=3))


# =============================
# LGBM (no OHE needed, handles categoricals internally)
# =============================

# Start from final; remove IDs/dates
drop_cols = ["CustomerID","ChurnStatus","first_txn_dt","last_txn_dt","last_cs_dt","LastLoginDate"]
X = final.drop(columns=[c for c in drop_cols if c in final.columns]).copy()
y = final["ChurnStatus"].astype(int)

# If previously OHE’d ServiceUsage, remove any 'channel_*' columns:
# Don't drop CustomerID yet - we need it for the merge
temp_drop_cols = [c for c in drop_cols if c in final.columns and c != "CustomerID"]
X = final.drop(columns=temp_drop_cols).copy()
# X = X.drop(columns=[c for c in X.columns if c.startswith("channel_")], errors="ignore")

# First check for ServiceUsage and merge if needed
if "ServiceUsage" not in X.columns and "ServiceUsage" in online.columns:
    X = X.merge(online[["CustomerID","ServiceUsage"]], left_on="CustomerID", right_on="CustomerID", how="left")

# NOW drop CustomerID after merge is complete
if "CustomerID" in X.columns:
    X = X.drop(columns=["CustomerID"])

# # Make sure 'ServiceUsage' exists exactly once; reattach from original sheet if missing
# if "ServiceUsage" not in X.columns and "ServiceUsage" in online.columns:
#     X = X.merge(online[["CustomerID","ServiceUsage"]], left_on="CustomerID", right_on="CustomerID", how="left")
#     if "CustomerID" in X.columns: X = X.drop(columns=["CustomerID"])

# Deduplicate names (safety)
X = dedupe_columns(X)

# Convert object to category dtype
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype("category")

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

lgbm = lgb.LGBMClassifier(
    n_estimators=1500, learning_rate=0.03, num_leaves=63,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    class_weight="balanced", random_state=42
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_test, y_test)], eval_metric="auc",
         categorical_feature=cat_cols)

p = lgbm.predict_proba(X_test)[:,1]
print("ROC-AUC:", f"{roc_auc_score(y_test, p):.3f}")
print("PR-AUC :", f"{average_precision_score(y_test, p):.3f}")
prec, rec, thr = precision_recall_curve(y_test, p)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1[:-1])]
print("Best threshold for LGBM:", best_thr)
print(classification_report(y_test, (p>=best_thr).astype(int), digits=3))

# =============================
# CatBoost (no OHE needed, handles categoricals internally)
# =============================

# 2) Make sure any remaining non-numeric (strings) are marked as categorical for CatBoost
# Include both object and category dtypes for CatBoost
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

model = CatBoostClassifier(
    depth=6, learning_rate=0.05, iterations=1500,
    loss_function="Logloss", eval_metric="AUC",
    class_weights=[1.0, (y==0).sum()/(y==1).sum()],
    random_seed=42, verbose=100
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)
p = model.predict_proba(test_pool)[:,1]

print("ROC-AUC:", f"{roc_auc_score(y_test, p):.3f}")
print("PR-AUC :", f"{average_precision_score(y_test, p):.3f}")

# Threshold tuning
prec, rec, thr = precision_recall_curve(y_test, p)
f1 = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1[:-1])]
y_hat = (p >= best_thr).astype(int)
print("Best threshold for CatBoost:", best_thr)
print(classification_report(y_test, y_hat, digits=3))

# ============================
# E) Feature importances
# ============================

# --------- LightGBM feature importance

# 1) As a tidy Series (by "gain" = total information gain)
gain_imp = pd.Series(lgbm.feature_importances_, index=X_train.columns, name="gain").sort_values(ascending=False)

# 2) If you prefer "split" counts instead of gain:
split_imp = pd.Series(lgbm.booster_.feature_importance(importance_type="split"),
                      index=X_train.columns, name="split").sort_values(ascending=False)

print("Top 15 (gain):\n", gain_imp.head(15))
print("\nTop 15 (split):\n", split_imp.head(15))

# Simple bar plot for top 20 by gain
topk = gain_imp.head(20)
plt.figure(figsize=(8,6))
topk.iloc[::-1].plot(kind="barh")
plt.title("LightGBM feature importance (gain)")
plt.tight_layout()
plt.show()

# --------- CatBoost feature importance
# Default importance (PredictionValuesChange)
imp_vals = model.get_feature_importance(train_pool, type="PredictionValuesChange")
cat_imp = pd.Series(imp_vals, index=X_train.columns).sort_values(ascending=False)

print("Top 15 (CatBoost - PredictionValuesChange):\n", cat_imp.head(15))

plt.figure(figsize=(8,6))
cat_imp.head(20).iloc[::-1].plot(kind="barh")
plt.title("CatBoost feature importance (PredictionValuesChange)")
plt.tight_layout()
plt.show()

# Alternative: LossFunctionChange
imp_loss = model.get_feature_importance(train_pool, type="LossFunctionChange")
cat_imp_loss = pd.Series(imp_loss, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 (CatBoost - LossFunctionChange):\n", cat_imp_loss.head(10))

# -------- SHAP for CatBoost
# SHAP summary values for a sample of rows
shap_vals = model.get_feature_importance(test_pool, type="ShapValues")
# shap_vals shape: (n_samples, n_features + 1), last column is expected value
shap_contrib = shap_vals[:, :-1]
shap_mean_abs = pd.Series(abs(shap_contrib).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
print("\nTop 10 by mean |SHAP|:\n", shap_mean_abs.head(10))

# summary plot (be patient, can be slow)
shap.summary_plot(shap_contrib, X_test, plot_type="dot")

# -------- RF feature importance
rf_imp = pd.Series(rf.feature_importances_, index=df_model.columns).sort_values(ascending=False)
print("\nTop 15 (RF - MeanDecreaseImpurity):\n", rf_imp.head(15))
plt.figure(figsize=(8,6))
rf_imp.head(20).iloc[::-1].plot(kind="barh")
plt.title("Random Forest feature importance (MDI)")
plt.tight_layout()
plt.show()

# Alternative: LGBM permutation importance (but this is agnostic???)
perm = permutation_importance(lgbm, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\nTop 15 (LGBM - Permutation) (AUC drop):\n", perm_imp.head(15))
plt.figure(figsize=(8,6))
perm_imp.head(20).iloc[::-1].plot(kind="barh")
plt.title("LGBM feature importance (permutation)")
plt.tight_layout()
plt.show()

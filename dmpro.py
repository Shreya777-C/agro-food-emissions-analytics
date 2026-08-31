

import os, math, json, argparse
import numpy as np
import pandas as pd

#for PNG heatmap
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# Metrics
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)

# Classification-style F1 via tertile bins (for a continuous target)
def _conf_counts(y_true, y_pred, cls):
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    return tp, fp, fn

def f1_macro_weighted_from_bins(y_true_bins, y_pred_bins):
    classes = np.unique(y_true_bins)
    f1s, supports = [], []
    for c in classes:
        tp, fp, fn = _conf_counts(y_true_bins, y_pred_bins, c)
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        f1s.append(f1)
        supports.append(np.sum(y_true_bins == c))
    macro = float(np.mean(f1s)) if len(f1s) else 0.0
    weighted = float(
        np.sum(np.array(f1s) * np.array(supports)) / (np.sum(supports) + 1e-12)
    ) if supports else 0.0
    return macro, weighted

def tertile_edges(y_train):
    q1, q2 = np.quantile(y_train, [1/3, 2/3])
    return np.array([-np.inf, q1, q2, np.inf], dtype=float)

def apply_bins(edges, y):
    return np.digitize(y, edges) - 1  #  0/1/2


# Train/Val/Test split

def train_val_test_split(X, y, test_size=0.2, val_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    remain = idx[n_test:]
    n_val = int(len(remain) * val_size / (1 - test_size))
    val_idx = remain[:n_val]
    train_idx = remain[n_val:]
    return (X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
            X.iloc[val_idx].reset_index(drop=True),
            y.iloc[val_idx].reset_index(drop=True),
            X.iloc[test_idx].reset_index(drop=True),
            y.iloc[test_idx].reset_index(drop=True))


# Internal CART used by RF and GBDT

class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")
    def __init__(self):
        self.feature = -1
        self.threshold = 0.0
        self.left = -1
        self.right = -1
        self.value = 0.0
        self.is_leaf = False

class _CARTRegressor:
    def __init__(self, max_depth=5, min_samples_split=4, min_samples_leaf=2,
                 max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.nodes = []
        self.n_features_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.nodes = []
        self._build(X, y, 0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._pred_row(row) for row in X], dtype=float)

    def _pred_row(self, row):
        i = 0
        while True:
            node = self.nodes[i]
            if node.is_leaf:
                return node.value
            i = node.left if row[node.feature] <= node.threshold else node.right

    def _build(self, X, y, depth):
        i = len(self.nodes); self.nodes.append(_TreeNode())
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            np.unique(y).size == 1):
            self.nodes[i].is_leaf = True
            self.nodes[i].value = float(np.mean(y))
            return i

        feat_idxs = np.arange(self.n_features_)
        if self.max_features is not None:
            m = min(self.max_features, self.n_features_)
            feat_idxs = self.rng.choice(self.n_features_, size=m, replace=False)

        parent_var = np.var(y) * len(y)
        best_feat, best_thr, best_gain = None, None, -np.inf

        for f in feat_idxs:
            x = X[:, f]
            vals = np.unique(x)
            if len(vals) <= 1:
                continue
            thrs = (vals[:-1] + vals[1:]) / 2.0
            order = np.argsort(x)
            x_s, y_s = x[order], y[order]
            y_c = np.cumsum(y_s)
            y2_c = np.cumsum(y_s ** 2)
            for t in thrs:
                pos = np.searchsorted(x_s, t, side="right") - 1
                left_n = pos + 1
                right_n = len(y) - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left_sum, left_sq = y_c[pos], y2_c[pos]
                right_sum, right_sq = y_c[-1] - left_sum, y2_c[-1] - left_sq
                left_var = left_sq - (left_sum ** 2) / left_n
                right_var = right_sq - (right_sum ** 2) / right_n
                gain = parent_var - (left_var + right_var)
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, t

        if best_feat is None or best_gain <= 1e-12:
            self.nodes[i].is_leaf = True
            self.nodes[i].value = float(np.mean(y))
            return i

        mask = X[:, best_feat] <= best_thr
        Xl, yl = X[mask], y[mask]
        Xr, yr = X[~mask], y[~mask]
        if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
            self.nodes[i].is_leaf = True
            self.nodes[i].value = float(np.mean(y))
            return i

        self.nodes[i].feature = best_feat
        self.nodes[i].threshold = float(best_thr)
        left_idx = self._build(Xl, yl, depth + 1)
        right_idx = self._build(Xr, yr, depth + 1)
        self.nodes[i].left, self.nodes[i].right = left_idx, right_idx
        return i


# 1) RandomForestCO2

class RandomForestCO2:
    def __init__(self, n_estimators=300, max_depth=10, min_samples_split=4, min_samples_leaf=2,
                 max_features="sqrt", bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []

    def _calc_max_features(self, d):
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(math.log2(d)))
        if isinstance(self.max_features, int):
            return min(d, self.max_features)
        return d

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.trees_ = []
        m_features = self._calc_max_features(d)
        for _ in range(self.n_estimators):
            idx = self.rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            Xi, yi = X[idx], y[idx]
            tree = _CARTRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=m_features,
                random_state=int(self.rng.integers(0, 1_000_000))
            ).fit(Xi, yi)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        return np.mean(preds, axis=1)


# 2) XGBStyleGBDTRegressor

class XGBStyleGBDTRegressor:
    def __init__(self, n_estimators=150, learning_rate=0.05, max_depth=3,
                 min_samples_split=4, min_samples_leaf=2,
                 subsample=0.9, colsample=0.9, random_state=7):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []
        self.feature_indices_ = []
        self.init_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.trees_, self.feature_indices_ = [], []
        self.init_ = float(np.mean(y))
        y_pred = np.full(n, self.init_, dtype=float)

        for m in range(self.n_estimators):
            if (m + 1) % 25 == 0:
                print(f"[XGB] round {m+1}/{self.n_estimators}")
            residual = y - y_pred
            rows = np.arange(n)
            if self.subsample < 1.0:
                k = max(10, int(self.subsample * n))
                rows = self.rng.choice(rows, size=k, replace=False)
            cols = np.arange(d)
            if self.colsample < 1.0:
                t = max(1, int(self.colsample * d))
                cols = np.sort(self.rng.choice(cols, size=t, replace=False))
            tree = _CARTRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                random_state=int(self.rng.integers(0, 1_000_000))
            ).fit(X[rows][:, cols], residual[rows])
            self.trees_.append(tree)
            self.feature_indices_.append(cols)
            y_pred += self.learning_rate * tree.predict(X[:, cols])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for t, cols in zip(self.trees_, self.feature_indices_):
            pred += self.learning_rate * t.predict(X[:, cols])
        return pred


# 3) Linear OLS with Year and Country FE and lag features

class OLSFixedEffects:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta_ = None

    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X

    def fit(self, X, y):
        Xb = self._add_intercept(X)
        self.beta_ = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)
        return self

    def predict(self, X):
        Xb = self._add_intercept(X)
        return Xb @ self.beta_


# Preprocess / feature engineering

TARGET = "total_emission"
COUNTRY_COL = "Area"
YEAR_COL = "Year"

# sector grouping heuristics for reporting
def sector_of(feature_name: str) -> str:
    f = feature_name.lower()
    m = {
        "fertilizer": "Fertilizers",
        "fertilizers": "Fertilizers",
        "manure": "Livestock/Manure",
        "livestock": "Livestock/Manure",
        "rice": "Rice Cultivation",
        "forestland": "Land/Forestry",
        "forest": "Land/Forestry",
        "net_forest": "Land/Forestry",
        "fires": "Fires",
        "on-farm": "On-farm Energy",
        "on_farm": "On-farm Energy",
        "energy": "On-farm Energy",
        "processing": "Food Processing",
        "transport": "Food Transport",
        "packaging": "Food Packaging",
        "household": "Food Consumption",
        "retail": "Food Retail",
        "urban": "Urban/Rural/Population",
        "rural": "Urban/Rural/Population",
        "population": "Urban/Rural/Population",
        "average_temperature": "Climate",
        "temperature": "Climate",
        "ippu": "IPPU",
        "drained_organic_soils": "Soils",
        "crop_residues": "Crop Residues",
    }
    for k, v in m.items():
        if k in f:
            return v
    if "__" in feature_name:  
        return "Country FE"
    if feature_name.lower() == YEAR_COL.lower():
        return "Year"
    return "Other"

def load_and_preprocess(csv_path, use_lags=False, lag=1):
    df = pd.read_csv(csv_path)
    # cleaning column names
    df.columns = [
        c.strip().replace("\u00a0", " ").replace("  ", " ")
         .replace("/", "_").replace("(", "").replace(")", "")
         .replace("-", "_").replace(" ", "_")
        for c in df.columns
    ]
    if TARGET not in df.columns or COUNTRY_COL not in df.columns or YEAR_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TARGET}', '{COUNTRY_COL}', '{YEAR_COL}'")

    # basic imputation
    for c in df.columns:
        if c == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # constructing lag features (t-1) 
    if use_lags:
        lag_cols = [TARGET]
        for cand in [
            "Fertilizers_Manufacturing",
            "Food_Processing",
            "Food_Transport",
            "Manure_applied_to_Soils",
            "Manure_Management",
            "Rice_Cultivation",
        ]:
            if cand in df.columns:
                lag_cols.append(cand)
        df = df.sort_values([COUNTRY_COL, YEAR_COL]).reset_index(drop=True)
        for c in lag_cols:
            df[f"{c}_lag{lag}"] = df.groupby(COUNTRY_COL)[c].shift(lag)
        df = df.dropna(subset=[f"{c}_lag{lag}" for c in lag_cols]).reset_index(drop=True)

    # one-hot encode country (droping the baseline for FE)
    countries = sorted(df[COUNTRY_COL].astype(str).unique().tolist())
    baseline = countries[0] if countries else None
    df_ohe = pd.get_dummies(df[COUNTRY_COL].astype(str), prefix=COUNTRY_COL)
    if baseline and f"{COUNTRY_COL}_{baseline}" in df_ohe.columns:
        df_ohe = df_ohe.drop(columns=[f"{COUNTRY_COL}_{baseline}"])

    # assembling X/Y
    numeric_cols = []
    if pd.api.types.is_numeric_dtype(df[YEAR_COL]):
        numeric_cols.append(YEAR_COL)
    for c in df.columns:
        if c in (TARGET, COUNTRY_COL, YEAR_COL):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    X = pd.concat([df[numeric_cols].astype(float), df_ohe.astype(float)], axis=1)
    y = df[TARGET].astype(float)
    feature_names = list(X.columns)
    return X, y, df, feature_names


# Permutation importance

def permutation_importance(model, X, y, feature_names, n_repeats=2, seed=42):
 
    rng = np.random.default_rng(seed)
    X_arr = np.asarray(X, dtype=float)
    base = rmse(y, model.predict(X_arr))
    p = X_arr.shape[1]
    imps = np.zeros(p)
    for j in range(p):
        scores = []
        for _ in range(n_repeats):
            Xp = X_arr.copy()
            rng.shuffle(Xp[:, j])
            scores.append(rmse(y, model.predict(Xp)))
        imps[j] = float(np.mean(scores) - base)
    df = pd.DataFrame({"feature": feature_names, "rmse_increase": imps})
    df = df.sort_values("rmse_increase", ascending=False).reset_index(drop=True)
    return df

def group_importance(df_imp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_imp.iterrows():
        rows.append((sector_of(str(r["feature"])), float(r["rmse_increase"])))
    g = (pd.DataFrame(rows, columns=["sector", "rmse_increase"])
           .groupby("sector", as_index=False)["rmse_increase"].sum())
    return g.sort_values("rmse_increase", ascending=False).reset_index(drop=True)


# World heatmap

def make_world_heatmap(df_raw, outdir):
    if not {"Area", "Year", TARGET}.issubset(df_raw.columns):
        return False
    latest = int(df_raw["Year"].max())
    d = df_raw[df_raw["Year"] == latest][["Area", TARGET]].copy()
    d.rename(columns={"Area": "country", TARGET: "total_emission"}, inplace=True)
    d.to_csv(os.path.join(outdir, "world_heatmap_data.csv"), index=False)
    if not PLOTLY_OK:
        print("Plotly not available so saved world_heatmap_data.csv instead.")
        return False
    fig = px.choropleth(
        d,
        locations="country",
        locationmode="country names",
        color="total_emission",
        color_continuous_scale="YlOrRd",
        title=f"Agri-Food CO₂ — Total Emissions (Latest year {latest})",
    )
    try:
        fig.write_image(os.path.join(outdir, "world_heatmap.png"), scale=2)
        return True
    except Exception:
        fig.write_html(os.path.join(outdir, "world_heatmap.html"))
        print("Could not save PNG do saved HTML instead.")
        return False


# Consensus across models

def build_consensus(outdir):
    paths = [
        ("LinearOLS", os.path.join(outdir, "feature_importance_LinearOLS.csv")),
        ("RandomForestCO2", os.path.join(outdir, "feature_importance_RandomForestCO2.csv")),
        ("XGBStyleGBDTRegressor", os.path.join(outdir, "feature_importance_XGBStyleGBDTRegressor.csv")),
    ]
    dfs = []
    for model, p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["model"] = model
            df["rank"] = df["rmse_increase"].rank(ascending=False, method="dense")
            dfs.append(df[["feature", "rmse_increase", "model", "rank"]])
    if not dfs:
        return
    M = pd.concat(dfs, ignore_index=True)
    agg = (
        M.assign(in_top20=M["rank"] <= 20)
         .groupby("feature", as_index=False)
         .agg(
             mean_rank=("rank", "mean"),
             mean_rmse_increase=("rmse_increase", "mean"),
             appear_top20=("in_top20", "sum"),
             models_seen=("model", "nunique"),
         )
         .sort_values(["appear_top20", "mean_rank"], ascending=[False, True])
    )
    agg.to_csv(os.path.join(outdir, "global_feature_consensus.csv"), index=False)

    grouped_paths = []
    for model, p in paths:
        gfile = os.path.join(outdir, f"feature_importance_grouped_{model}.csv")
        if os.path.exists(gfile):
            grouped_paths.append((model, gfile))
        elif os.path.exists(p):
            tmp = group_importance(pd.read_csv(p))
            tmp.to_csv(gfile, index=False)
            grouped_paths.append((model, gfile))

    if grouped_paths:
        G = []
        for model, gp in grouped_paths:
            dd = pd.read_csv(gp)
            dd["model"] = model
            dd["rank"] = dd["rmse_increase"].rank(ascending=False, method="dense")
            G.append(dd)
        GM = pd.concat(G, ignore_index=True)
        gagg = (
            GM.groupby("sector", as_index=False)
              .agg(
                  mean_rank=("rank", "mean"),
                  mean_rmse_increase=("rmse_increase", "mean"),
                  models_seen=("model", "nunique"),
              )
              .sort_values(["mean_rmse_increase"], ascending=False)
        )
        gagg.to_csv(os.path.join(outdir, "global_grouped_consensus.csv"), index=False)


# Cross-validation and Hyperparameter Tuning

def cross_validate_model(model_name, model_ctor, X, y, param_grid, k=5, outdir=None, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    rows = []
    best_rmse = None
    best_params = None

    for params in param_grid:
        fold_rmses = []
        params_str = json.dumps(params, sort_keys=True)
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            X_tr = X.values[train_idx]
            y_tr = y.values[train_idx]
            X_te = X.values[test_idx]
            y_te = y.values[test_idx]

            model = model_ctor(**params).fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            rm = rmse(y_te, y_pred)
            ma = mae(y_te, y_pred)
            r2s = r2_score(y_te, y_pred)
            edges = tertile_edges(y_tr)
            tb, pb = apply_bins(edges, y_te), apply_bins(edges, y_pred)
            f1M, f1W = f1_macro_weighted_from_bins(tb, pb)

            rows.append({
                "model": model_name,
                "params": params_str,
                "fold": i,
                "rmse": rm,
                "mae": ma,
                "r2": r2s,
                "f1_macro": f1M,
                "f1_weighted": f1W,
            })
            fold_rmses.append(rm)

        avg_rmse = float(np.mean(fold_rmses))
        if best_rmse is None or avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params

    df_cv = pd.DataFrame(rows)
    if outdir is not None:
        df_cv.to_csv(os.path.join(outdir, f"cv_results_{model_name}.csv"), index=False)

    return best_params

def tune_random_forest(X, y, outdir):
    # smaller grid for speed
    param_grid = []
    for n_est in [80, 150]:
        for depth in [6, 8]:
            for mf in ["sqrt"]:
                param_grid.append({
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "max_features": mf,
                    "min_samples_split": 4,
                    "min_samples_leaf": 2,
                    "bootstrap": True,
                    "random_state": 42,
                })
    best = cross_validate_model(
        "RandomForestCO2",
        lambda **kw: RandomForestCO2(**kw),
        X, y,
        param_grid,
        k=5,
        outdir=outdir,
    )
    if best is not None:
        with open(os.path.join(outdir, "best_params_RandomForestCO2.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
    return best

def tune_xgb_gbdt(X, y, outdir):
    # smaller grid for speed
    param_grid = []
    for n_est in [80, 120]:
        for lr in [0.05, 0.1]:
            for depth in [2, 3]:
                param_grid.append({
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "max_depth": depth,
                    "min_samples_split": 4,
                    "min_samples_leaf": 2,
                    "subsample": 0.9,
                    "colsample": 0.9,
                    "random_state": 7,
                })
    best = cross_validate_model(
        "XGBStyleGBDTRegressor",
        lambda **kw: XGBStyleGBDTRegressor(**kw),
        X, y,
        param_grid,
        k=5,
        outdir=outdir,
    )
    if best is not None:
        with open(os.path.join(outdir, "best_params_XGBStyleGBDTRegressor.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
    return best

# Forecasting and Scenario Analysis

def forecast_future(models, df_raw, feature_names, outdir,
                    horizon=5, use_lags=False, lag=1):
    
    if YEAR_COL not in df_raw.columns or COUNTRY_COL not in df_raw.columns:
        return

    latest_year = int(df_raw[YEAR_COL].max())

    # scenarios: baseline and simple policy shocks
    scenarios = {
        "baseline": {},
        "low_fertilizer": {"Fertilizers_Manufacturing": 0.9},
        "high_transport": {"Food_Transport": 1.2},
    }

    lag_suffix = f"_lag{lag}"
    lag_cols = [c for c in df_raw.columns if c.endswith(lag_suffix)]
    records = []

    for scenario_name, scales in scenarios.items():
        for country, df_c in df_raw.groupby(COUNTRY_COL):
            df_c = df_c.sort_values(YEAR_COL)
            last_row = df_c.iloc[-1]
            last_total = float(last_row[TARGET])

            #  (non-lag) numeric drivers, scaled
            base_vals = {}
            for col in df_raw.columns:
                if col in (TARGET, COUNTRY_COL, YEAR_COL):
                    continue
                if col.endswith(lag_suffix):
                    continue
                if not pd.api.types.is_numeric_dtype(df_raw[col]):
                    continue
                val = float(last_row[col])
                # applying scenario scale if defined for the base variable
                if col in scales:
                    val *= scales[col]
                base_vals[col] = val

            for step in range(1, horizon + 1):
                year = latest_year + step

                # lag values
                lag_vals = {}
                if use_lags and lag == 1:
                    # target lag: previous year's emission
                    lag_vals[f"{TARGET}_lag{lag}"] = last_total
                    for col in lag_cols:
                        if col == f"{TARGET}_lag{lag}":
                            continue
                        base_name = col[:-len(lag_suffix)]
                        base_val = base_vals.get(
                            base_name,
                            float(last_row.get(base_name, 0.0)),
                        )
                        if base_name in scales:
                            base_val *= scales[base_name]
                        lag_vals[col] = base_val

                # building feature row for this country/year
                feat_row = {}
                for feat in feature_names:
                    if feat == YEAR_COL:
                        feat_row[feat] = float(year)
                    elif feat in base_vals:
                        feat_row[feat] = base_vals[feat]
                    elif use_lags and feat in lag_vals:
                        feat_row[feat] = lag_vals[feat]
                    elif feat.startswith(COUNTRY_COL + "_"):
                        cc = feat[len(COUNTRY_COL) + 1:]
                        feat_row[feat] = 1.0 if str(country) == cc else 0.0
                    else:
                        if feat in df_raw.columns:
                            feat_row[feat] = float(last_row[feat])
                        else:
                            feat_row[feat] = 0.0

                X_row = np.array([[feat_row[f] for f in feature_names]], dtype=float)

                # predictions per model
                for model_name, model in models.items():
                    yhat = float(model.predict(X_row)[0])
                    records.append({
                        "scenario": scenario_name,
                        "model": model_name,
                        "Area": country,
                        "Year": year,
                        "predicted_total_emission": yhat,
                    })

                
                if "XGBStyleGBDTRegressor" in models:
                    last_total = float(models["XGBStyleGBDTRegressor"].predict(X_row)[0])
                else:
                    last_total = float(models[list(models.keys())[0]].predict(X_row)[0])

    if not records:
        return

    df_fore = pd.DataFrame(records)
    df_fore.to_csv(os.path.join(outdir, "forecasts_raw.csv"), index=False)

    # global trends
    (df_fore.groupby(["scenario", "model", "Year"], as_index=False)["predicted_total_emission"]
            .sum()
            .rename(columns={"predicted_total_emission": "global_predicted_total_emission"})
            .to_csv(os.path.join(outdir, "forecasts_global_trend.csv"), index=False))

    # per-scenario CSVs
    for scen in df_fore["scenario"].unique():
        df_fore[df_fore["scenario"] == scen].to_csv(
            os.path.join(outdir, f"scenario_{scen}.csv"), index=False
        )


# Main training and eval routine

def run(csv_path, use_lags=False, lag=1, outdir="results",
        rf_params=None, xgb_params=None,
        do_forecast=True, forecast_horizon=5):
    os.makedirs(outdir, exist_ok=True)

    # 1) Loading and  preprocess
    print(" Loading and preprocessing data...")
    X, y, df_raw, feature_names = load_and_preprocess(
        csv_path, use_lags=use_lags, lag=lag
    )
    Xtr, ytr, Xval, yval, Xte, yte = train_val_test_split(
        X, y, test_size=0.2, val_size=0.2, random_state=42
    )
    edges = tertile_edges(ytr.values)

    rows_metrics = []

    # 2) Linear OLS
    print(" Fitting Linear OLS...")
    t0 = pd.Timestamp.now()
    lin = OLSFixedEffects(fit_intercept=True).fit(Xtr.values, ytr.values)
    pred_lin = lin.predict(Xte.values)
    m_rmse = rmse(yte, pred_lin)
    m_mae = mae(yte, pred_lin)
    m_r2 = r2_score(yte, pred_lin)
    tb, pb = apply_bins(edges, yte.values), apply_bins(edges, pred_lin)
    f1M, f1W = f1_macro_weighted_from_bins(tb, pb)
    dt = (pd.Timestamp.now() - t0).total_seconds()
    rows_metrics.append(("LinearOLS", m_rmse, m_mae, m_r2, f1M, f1W, dt))

    print(" Computing Linear OLS permutation importance...")
    imp_lin = permutation_importance(
        lin, Xte.values, yte.values, feature_names, n_repeats=2, seed=42
    )
    imp_lin.to_csv(os.path.join(outdir, "feature_importance_LinearOLS.csv"), index=False)
    group_importance(imp_lin).to_csv(
        os.path.join(outdir, "feature_importance_grouped_LinearOLS.csv"), index=False
    )
    print(
        f"LinearOLS -> RMSE={m_rmse:.4f} | MAE={m_mae:.4f} | R2={m_r2:.4f} "
        f"| F1_macro={f1M:.4f} | F1_weighted={f1W:.4f}  ({dt:.1f}s)"
    )

    # 3) RandomForestCO2
    print(">>> Fitting RandomForest...")
    t0 = pd.Timestamp.now()
    rf_kwargs = dict(
        n_estimators=80,      # lighter than 300
        max_depth=6,         # lighter than 10
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    if rf_params is not None:
        rf_kwargs.update(rf_params)

    rf = RandomForestCO2(**rf_kwargs).fit(Xtr.values, ytr.values)
    pred_rf = rf.predict(Xte.values)
    m_rmse = rmse(yte, pred_rf)
    m_mae = mae(yte, pred_rf)
    m_r2 = r2_score(yte, pred_rf)
    tb, pb = apply_bins(edges, yte.values), apply_bins(edges, pred_rf)
    f1M, f1W = f1_macro_weighted_from_bins(tb, pb)
    dt = (pd.Timestamp.now() - t0).total_seconds()
    rows_metrics.append(("RandomForestCO2", m_rmse, m_mae, m_r2, f1M, f1W, dt))

    print(">>> Computing RF permutation importance...")
    imp_rf = permutation_importance(
        rf, Xte.values, yte.values, feature_names, n_repeats=2, seed=42
    )
    imp_rf.to_csv(os.path.join(outdir, "feature_importance_RandomForestCO2.csv"), index=False)
    group_importance(imp_rf).to_csv(
        os.path.join(outdir, "feature_importance_grouped_RandomForestCO2.csv"), index=False
    )
    print(
        f"RandomForestCO2 -> RMSE={m_rmse:.4f} | MAE={m_mae:.4f} | R2={m_r2:.4f} "
        f"| F1_macro={f1M:.4f} | F1_weighted={f1W:.4f}  ({dt:.1f}s)"
    )

    # 4) XGB-style GBDT
    print(">>> Fitting XGB GBDT...")
    t0 = pd.Timestamp.now()
    xgb_kwargs = dict(
        n_estimators=80,      # lighter than 150
        learning_rate=0.07,  
        max_depth=3,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.9,
        colsample=0.9,
        random_state=7,
    )
    if xgb_params is not None:
        xgb_kwargs.update(xgb_params)

    xgb = XGBStyleGBDTRegressor(**xgb_kwargs).fit(Xtr.values, ytr.values)
    pred_xgb = xgb.predict(Xte.values)
    m_rmse = rmse(yte, pred_xgb)
    m_mae = mae(yte, pred_xgb)
    m_r2 = r2_score(yte, pred_xgb)
    tb, pb = apply_bins(edges, yte.values), apply_bins(edges, pred_xgb)
    f1M, f1W = f1_macro_weighted_from_bins(tb, pb)
    dt = (pd.Timestamp.now() - t0).total_seconds()
    rows_metrics.append(("XGBStyleGBDTRegressor", m_rmse, m_mae, m_r2, f1M, f1W, dt))

    print(">>> Computing XGB permutation importance...")
    imp_xgb = permutation_importance(
        xgb, Xte.values, yte.values, feature_names, n_repeats=2, seed=42
    )
    imp_xgb.to_csv(
        os.path.join(outdir, "feature_importance_XGBStyleGBDTRegressor.csv"),
        index=False,
    )
    group_importance(imp_xgb).to_csv(
        os.path.join(outdir, "feature_importance_grouped_XGBStyleGBDTRegressor.csv"),
        index=False,
    )
    print(
        f"XGBStyleGBDTRegressor -> RMSE={m_rmse:.4f} | MAE={m_mae:.4f} | R2={m_r2:.4f} "
        f"| F1_macro={f1M:.4f} | F1_weighted={f1W:.4f}  ({dt:.1f}s)"
    )

    # 5) Saving the metrics table
    dfm = pd.DataFrame(
        rows_metrics,
        columns=["model", "rmse", "mae", "r2", "f1_macro", "f1_weighted", "seconds"],
    )
    dfm.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # 6) KPI summaries from raw data
    total = float(df_raw[TARGET].sum())
    with open(os.path.join(outdir, "total_emissions.txt"), "w", encoding="utf-8") as f:
        f.write(f"{total:,.3f}\n")

    (df_raw.groupby(COUNTRY_COL)[TARGET].mean()
          .reset_index(name="avg_emission")
          .sort_values("avg_emission", ascending=False)
          .to_csv(os.path.join(outdir, "avg_emissions_per_country.csv"), index=False))

    latest_year = int(df_raw[YEAR_COL].max())
    top_row = (
        df_raw[df_raw[YEAR_COL] == latest_year]
        .sort_values(TARGET, ascending=False)
        .iloc[0]
    )
    with open(os.path.join(outdir, "top_emitter_latest_year.txt"), "w", encoding="utf-8") as f:
        f.write(f"{top_row[COUNTRY_COL]} — {top_row[TARGET]:,.3f} ({latest_year})\n")

    # 7) World heatmap
    print(" Building world heatmap...")
    make_world_heatmap(df_raw, outdir)

    # 8) Cross-model consensus and sector summaries
    print(" Building consensus summaries...")
    build_consensus(outdir)

    # 9) Forecasting and scenarios (
    if do_forecast:
        print("Running forecasting and scenario analysis...")
        forecast_future(
            models={
                "LinearOLS": lin,
                "RandomForestCO2": rf,
                "XGBStyleGBDTRegressor": xgb,
            },
            df_raw=df_raw,
            feature_names=feature_names,
            outdir=outdir,
            horizon=forecast_horizon,
            use_lags=use_lags,
            lag=lag,
        )

    print("\nSaved to:", os.path.abspath(outdir))
    for fn in [
        "metrics.csv",
        "feature_importance_LinearOLS.csv",
        "feature_importance_RandomForestCO2.csv",
        "feature_importance_XGBStyleGBDTRegressor.csv",
        "feature_importance_grouped_LinearOLS.csv",
        "feature_importance_grouped_RandomForestCO2.csv",
        "feature_importance_grouped_XGBStyleGBDTRegressor.csv",
        "global_feature_consensus.csv",
        "global_grouped_consensus.csv",
        "avg_emissions_per_country.csv",
        "total_emissions.txt",
        "top_emitter_latest_year.txt",
        "world_heatmap.png",
        "world_heatmap.html",
        "world_heatmap_data.csv",
        "cv_results_RandomForestCO2.csv",
        "cv_results_XGBStyleGBDTRegressor.csv",
        "best_params_RandomForestCO2.json",
        "best_params_XGBStyleGBDTRegressor.json",
        "forecasts_raw.csv",
        "forecasts_global_trend.csv",
        "scenario_baseline.csv",
        "scenario_low_fertilizer.csv",
        "scenario_high_transport.csv",
    ]:
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            print(" -", fn)


# CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to Agrofood CSV (e.g., Agri.csv)")
    ap.add_argument("--use_lags", action="store_true",
                    help="Add t-1 lags for selected columns")
    ap.add_argument("--lag", type=int, default=1,
                    help="Lag years (default 1)")
    ap.add_argument("--outdir", type=str, default="results",
                    help="Output folder (default: results)")
    ap.add_argument("--do_cv", action="store_true",
                    help="Run 5-fold CV + tuning for RF and XGB before main training")
    ap.add_argument("--no_forecast", action="store_true",
                    help="Disable future forecasting and scenario analysis")
    ap.add_argument("--forecast_horizon", type=int, default=5,
                    help="Years to forecast beyond last year (default 5)")
    args = ap.parse_args()

    rf_params = None
    xgb_params = None

    if args.do_cv:
        print("Running cross-validation & tuning...")
        X_all, y_all, _, _ = load_and_preprocess(
            args.csv, use_lags=args.use_lags, lag=args.lag
        )
        rf_params = tune_random_forest(X_all, y_all, outdir=args.outdir)
        xgb_params = tune_xgb_gbdt(X_all, y_all, outdir=args.outdir)
        print("Tuning complete.")
        print("Best RF params:", rf_params)
        print("Best XGB params:", xgb_params)

    run(
        csv_path=args.csv,
        use_lags=args.use_lags,
        lag=args.lag,
        outdir=args.outdir,
        rf_params=rf_params,
        xgb_params=xgb_params,
        do_forecast=not args.no_forecast,
        forecast_horizon=args.forecast_horizon,
    )

if __name__ == "__main__":
    main()

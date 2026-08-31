# app.py — Agri-Food CO2 Dashboard (Flask)
# Run:  conda activate agri  &&  python app.py
import os, subprocess, json
import pandas as pd
from flask import (
    Flask, render_template, request, send_from_directory,
    redirect, url_for, flash
)

# ---------- paths pinned to this folder ----------
BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
DM_PATH        = os.path.join(BASE_DIR, "dmpro.py")
CSV_DEFAULT    = os.path.join(BASE_DIR, "Agri.csv")
LAST_CSV_TXT   = os.path.join(RESULTS_DIR, "last_csv_path.txt")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = "dev-secret"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- helpers ----------
def safe_read_csv(path, empty_cols=None):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=empty_cols or [])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=empty_cols or [])

def get_last_csv():
    if os.path.exists(LAST_CSV_TXT):
        try:
            return open(LAST_CSV_TXT, "r", encoding="utf-8").read().strip()
        except Exception:
            pass
    return CSV_DEFAULT

def set_last_csv(p):
    try:
        with open(LAST_CSV_TXT, "w", encoding="utf-8") as f:
            f.write(p)
    except Exception:
        pass

# ---------- routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    # Recompute pipeline
    if request.method == "POST":
        csv_path = request.form.get("csv_path", get_last_csv()).strip()
        use_lags = "use_lags" in request.form
        lag      = request.form.get("lag", "1").strip() or "1"

        if not os.path.exists(csv_path):
            flash(f"CSV not found: {csv_path}", "danger")
            return redirect(url_for("index"))
        if not os.path.exists(DM_PATH):
            flash(f"dmpro.py not found at: {DM_PATH}", "danger")
            return redirect(url_for("index"))

        cmd = ["python", DM_PATH, "--csv", csv_path]
        if use_lags:
            cmd += ["--use_lags", "--lag", lag]

        try:
            # run in this folder so outputs land in BASE_DIR/results
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            with open(os.path.join(RESULTS_DIR, "last_run_log.txt"), "w", encoding="utf-8") as f:
                f.write(out)
            if proc.returncode != 0:
                flash("Recompute failed. Click 'View last log'.", "danger")
            else:
                set_last_csv(csv_path)  # remember CSV for all pages
                flash("Recompute finished.", "success")
        except Exception as e:
            flash(f"Failed to run dmpro.py: {e}", "danger")

        return redirect(url_for("index"))

    # GET — load artifacts for Overview
    csv_for_ui = get_last_csv()
    metrics = safe_read_csv(os.path.join(RESULTS_DIR, "metrics.csv"),
                            ["model","rmse","mae","r2","f1_macro","f1_weighted"])

    df_raw = safe_read_csv(csv_for_ui)
    total = avg_per_country = top_value = latest_year = None
    top_country = None

    if not df_raw.empty and {"Area","Year","total_emission"}.issubset(df_raw.columns):
        total = float(df_raw["total_emission"].sum())
        avg_per_country = float(df_raw.groupby("Area")["total_emission"].mean().mean())
        latest_year = int(df_raw["Year"].max())
        latest_df = df_raw[df_raw["Year"] == latest_year]
        if not latest_df.empty:
            row = latest_df.sort_values("total_emission", ascending=False).iloc[0]
            top_country, top_value = str(row["Area"]), float(row["total_emission"])

    heatmap_png  = os.path.exists(os.path.join(RESULTS_DIR, "world_heatmap.png"))
    heatmap_html = os.path.exists(os.path.join(RESULTS_DIR, "world_heatmap.html"))
    heatmap_csv  = os.path.exists(os.path.join(RESULTS_DIR, "world_heatmap_data.csv"))

    return render_template("index.html",
        csv_default=csv_for_ui.replace("\\", "\\\\"),
        metrics=metrics.to_dict(orient="records"),
        total=total, avg_per_country=avg_per_country,
        top_country=top_country, top_value=top_value, latest_year=latest_year,
        heatmap_png=heatmap_png, heatmap_html=heatmap_html, heatmap_csv=heatmap_csv
    )

@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

@app.route("/country")
def country():
    df_raw = safe_read_csv(get_last_csv())
    countries = sorted(df_raw["Area"].dropna().astype(str).unique()) if "Area" in df_raw.columns else []
    sel = request.args.get("name", countries[0] if countries else None)
    data = []
    if sel and not df_raw.empty and {"Area","Year","total_emission"}.issubset(df_raw.columns):
        sub = df_raw[df_raw["Area"].astype(str) == sel].sort_values("Year")
        data = sub[["Year","total_emission"]].to_dict(orient="records")
    return render_template("country.html", countries=countries, sel=sel, data=json.dumps(data))

@app.route("/features")
def features():
    model = request.args.get("model", "XGBStyleGBDTRegressor")
    fname = f"feature_importance_{model}.csv"
    df = safe_read_csv(os.path.join(RESULTS_DIR, fname), ["feature","rmse_increase"])
    rows = df.sort_values("rmse_increase", ascending=False).head(20).to_dict(orient="records")
    models = ["LinearOLS", "RandomForestCO2", "XGBStyleGBDTRegressor"]
    return render_template("features.html", model=model, models=models, rows=json.dumps(rows))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

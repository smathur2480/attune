"""
focus_core.reference -- population reference norms for EEG features, used ONLY when a user has no personal EEG baseline yet.

  simple score : population mean/SD (not adjusted for anything)
  complex score: age/sex/habit-adjusted expectation (this is where age, sex and habits enter the complex score)

Once a personal EEG baseline exists, the baseline already contains the user's own age/sex/habit effects, so no adjustment is applied.
The reference is fitted by `baseline.py reference` from TDBRAIN-derived features. A covariate model is used for a feature only if
it improves cross-validated R^2 (otherwise plain mean/SD); in our TDBRAIN run it rarely did, which is why age/sex matter little.

LITERATURE
  [vanDijk2022] van Dijk H, van Wingen G, Denys D, Olbrich S, van Ruth R, Arns M (2022). The two decades brainclinics research
                archive for insights in neurophysiology (TDBRAIN) database. Scientific Data 9:333.  (credit required by its DUA)
  [Arns2013]    Arns M, Conners CK, Kraemer HC (2013). A decade of EEG theta/beta ratio research in ADHD: a meta-analysis.
                J Atten Disord 17(5):374-383.  -> TBR varies with age; it does NOT reliably separate ADHD from controls, which is why
                the score treats it as a state marker relative to a person's own level, never as a diagnostic.
The output file holds only aggregate coefficients (no participant rows); check your data-use agreement before committing it.
"""
import numpy as np
import pandas as pd

REF_MAP = {"alpha_rel": "frontal_alpha", "tbr": "frontal_tbr", "beta_rel": "frontal_beta"}   # core feature <- TDBRAIN extractor column


def cov_vec(age_mu, age_sd, profile):
    """[z_age, z_age^2, female, uses_cigarette, uses_caffeine, uses_alcohol]  (use = hours_since given)."""
    age = float(profile.get("age", age_mu))
    z = (age - age_mu) / age_sd
    hrs = profile.get("hours_since", {}) or {}
    used = lambda k: 0.0 if hrs.get(k) is None else 1.0                                     # noqa: E731
    return np.array([z, z * z, 1.0 if str(profile.get("sex", "")).upper().startswith("F") else 0.0,
                     used("cigarette"), used("caffeine"), used("alcohol")])


def reference_expectation(ref, feature, profile):
    """(mean, sd, basis text). profile None -> plain population mean/SD; otherwise the age/sex/habit-matched expectation."""
    spec = ref["features"][feature]
    if profile is None:
        return spec["plain_mean"], max(spec["plain_sd"], 1e-6), "population reference (not age/sex adjusted)"
    pred = spec["coef"][0] + float(cov_vec(ref["age_mu"], ref["age_sd"], profile) @ np.array(spec["coef"][1:]))
    return pred, max(spec["sd"], 1e-6), "age/sex/habit-matched reference"


def _ridge(X, y, lam=1.0):
    Xd = np.c_[np.ones(len(X)), X]
    P = np.eye(Xd.shape[1]) * lam
    P[0, 0] = 0
    return np.linalg.solve(Xd.T @ Xd + P, Xd.T @ y)


def fit_reference(cohort_csv, features_csv, ref_group="HEALTHY", min_r2=0.0, seed=0):
    """Fit the reference from the cohort file (tdbrain_cohort_prep.py) and an EEG feature file (tdbrain_eeg_features.py)."""
    coh = pd.read_csv(cohort_csv)
    t = pd.read_csv(features_csv)
    if "status" in t.columns:
        t = t[t.status == "ok"]
    d = coh.merge(t, on=["TDBRAIN_ID", "sessID"], how="inner")
    first = d["sessID"] == d.groupby("TDBRAIN_ID")["sessID"].transform("min")
    ref = d[(d.group == ref_group) & first].reset_index(drop=True)
    use = lambda col: ((ref[f"{col}_99"] == 0) & (ref[f"{col}_miss"] == 0)).astype(float).to_numpy()   # noqa: E731
    mu, sd = float(ref.age.mean()), float(ref.age.std())
    z = (ref.age.to_numpy(float) - mu) / sd
    X = np.column_stack([z, z * z, ref.sex_F.to_numpy(float), use("cigarette"), use("coffee"), use("beer")])
    rng = np.random.default_rng(seed)
    out = dict(schema="focus-reference/2", ref_group=ref_group, n_ref=int(len(ref)), age_mu=mu, age_sd=sd, features={})
    log = []
    for feat, col in REF_MAP.items():
        if col not in ref.columns or ref[col].notna().sum() < 50:
            continue
        ok = ref[col].notna().to_numpy()
        y, Xo = ref.loc[ok, col].to_numpy(float), X[ok]
        oof, idx = np.zeros(len(y)), rng.permutation(len(y))
        for k in range(5):
            te = idx[k::5]
            tr = np.setdiff1d(idx, te)
            c = _ridge(Xo[tr], y[tr])
            oof[te] = c[0] + Xo[te] @ c[1:]
        sd_res = float(np.std(y - oof, ddof=1))
        r2 = 1 - sd_res ** 2 / float(np.var(y, ddof=1))
        used = r2 >= min_r2
        coef = _ridge(Xo, y) if used else np.r_[float(y.mean()), np.zeros(6)]
        out["features"][feat] = dict(coef=coef.tolist(), sd=sd_res if used else float(y.std(ddof=1)), plain_mean=float(y.mean()),
                                     plain_sd=float(y.std(ddof=1)), oof_r2=round(float(r2), 4), covariates_used=bool(used), n=int(ok.sum()))
        log.append(f"  {feat:10s} n={ok.sum():4d} out-of-sample R2 of age/sex/habits={r2:+.3f} -> "
                   f"{'covariate-adjusted' if used else 'plain mean/SD'}")
    return out, log

# Deploying the dashboard to Streamlit Community Cloud

This makes the dashboard a 24/7 public URL (with a password) instead of a
Streamlit process you start on your Mac. Three things have to be solved:
the model/data **artifacts** (not normally in git), the **password**, and
**ledger persistence** (the cloud filesystem is wiped on every redeploy).

Steps 1–2 you run on your **Mac**; steps 3–5 are in the Streamlit Cloud and
GitHub web UIs. The code/config for all of this is already on the branch.

---

## 1. Commit the runtime artifacts (Mac)

The app loads trained-model and processed-data files that are normally
gitignored (regenerable from Kaggle). The cloud has no Kaggle access, so
these must be in the repo. Force-add exactly these six:

```bash
cd ~/WorldCupPredictor
git checkout claude/ip-access-phone-issue-27sw97

git add -f \
  models/outcome_model.joblib \
  data/processed/matches.csv \
  data/processed/wc2026_fixtures_frozen.csv \
  data/processed/current_elo.csv \
  data/raw/market_odds_2026.csv \
  data/raw/shootouts.csv

git add reports/paper_ledger.csv        # your restored bets
git commit -m "Add runtime artifacts + ledger for cloud deploy"
```

> These are a few MB total — fine for git. They're a frozen snapshot; see
> "Updating results" below for keeping them current on the cloud.

## 2. Push the branch (Mac)

```bash
git push -u origin claude/ip-access-phone-issue-27sw97
```

(You can deploy straight from this branch; no need to merge to `main` yet.)

## 3. Create the Streamlit Cloud app (web)

1. Go to <https://share.streamlit.io> and sign in with the GitHub account
   that owns `kedarvyas/WorldCupPredictor`.
2. **Create app → Deploy a public app from a repo.**
3. Set:
   - **Repository:** `kedarvyas/WorldCupPredictor`
   - **Branch:** `claude/ip-access-phone-issue-27sw97`
   - **Main file path:** `app/dashboard.py`
4. Deploy. First build takes a few minutes (it installs `requirements.txt`).

## 4. Set the password (web)

In the app's **Settings → Secrets**, paste:

```toml
password = "choose-a-strong-password"
```

Reboot the app. It now asks for that password before showing anything.
(See `.streamlit/secrets.toml.example` for all available keys.)

## 5. Make logged bets survive redeploys (web, optional but recommended)

Without this, any bet you log on the cloud is **lost on the next redeploy** —
exactly the data-loss footgun we just hit. The app mirrors the ledger to a
private GitHub **gist** when these secrets are set:

1. Create a **secret gist** at <https://gist.github.com> with one file named
   exactly `paper_ledger.csv` — paste your current ledger CSV as its content.
2. Copy the **gist id** from the URL (the long hash).
3. Create a token with the **`gist`** scope at
   <https://github.com/settings/tokens> (classic + "gist", or a fine-grained
   token with Gists read/write).
4. Add to **Settings → Secrets**:

```toml
github_token = "ghp_..."
gist_id = "your_gist_id"
```

On each session the app pulls the gist; after each logged bet (or recorded
closing line) it pushes back. A gist is used instead of a repo commit so a
bet doesn't trigger an app redeploy.

---

## Known limitations on the cloud

- **"↻ Refresh results from data source"** needs Kaggle credentials and a
  writable disk — it won't work on the cloud as-is. Add `KAGGLE_USERNAME` /
  `KAGGLE_KEY` to secrets if you want it, but the refreshed data still won't
  persist across redeploys.
- **Manual result entry** writes to the (ephemeral) local disk, so entries
  are lost on redeploy. Fine within a session; for permanence, re-commit
  `data/processed/matches.csv` (step 1) or run the update on your Mac.
- **Updating results:** simplest path is to refresh on your Mac
  (`python -m src.data.download && python -m src.data.clean`), re-commit
  `data/processed/matches.csv`, and push — the cloud app redeploys
  automatically.

## Running locally (unchanged)

With no secrets file, none of the above activates: no password, ledger stays
in `reports/paper_ledger.csv`. `streamlit run app/dashboard.py` works exactly
as before.

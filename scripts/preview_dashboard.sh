#!/bin/bash
# Preview-server wrapper: maps the harness-assigned $PORT onto Streamlit's
# port flag so previews never collide with the production instance on 8501
# (the tailnet-facing server the phone bookmark points at).
exec .venv/bin/streamlit run app/dashboard.py \
  --server.headless true \
  --server.port "${PORT:-8503}"

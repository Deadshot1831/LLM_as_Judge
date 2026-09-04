.PHONY: install db dataset dataset-offline judge agreement disagreements bias label dashboard baseline gate test

install:      ; pip install -e ".[ci]"
db:           ; docker compose up -d && sleep 3 && python -m judge.db
dataset:      ; python scripts/make_dataset.py
dataset-offline: ; python scripts/make_dataset.py --offline
judge:        ; python -m judge.run_judge --rubric v2 --splits dev,test,bias
agreement:    ; python -m judge.agreement --split test --disagreements 20
bias:         ; python -m judge.bias --all --rubric v2
label:        ; streamlit run app/label.py
dashboard:    ; streamlit run app/dashboard.py
baseline:     ; python scripts/update_baseline.py --rubric v2
gate:         ; pytest tests/test_eval_gate.py -q
test:         ; pytest tests/test_analysis.py tests/test_ui_smoke.py -q

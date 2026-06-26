#!/bin/bash
set -euo pipefail

dataset="${1:-}"
if [[ -z "$dataset" ]]; then
  echo "usage: $0 <laptop|restaurant|twitter>" >&2
  exit 1
fi

case "$dataset" in
  laptop|restaurant|twitter) ;;
  *)
    echo "unsupported dataset: $dataset" >&2
    exit 1
    ;;
esac

find_best_teacher() {
  python3 - "$dataset" <<'PY'
from pathlib import Path
import re
import sys

dataset = sys.argv[1]
pattern = re.compile(rf"ssegcnbert_{dataset}_acc_(\d+\.\d+)_f1_(\d+\.\d+)$")
best = None
best_metrics = (-1.0, -1.0)

for path in Path("state_dict").glob(f"ssegcnbert_{dataset}_acc_*_f1_*"):
    match = pattern.fullmatch(path.name)
    if not match:
        continue
    metrics = tuple(float(x) for x in match.groups())
    if metrics > best_metrics:
        best = path
        best_metrics = metrics

if best is None:
    raise SystemExit(f"No teacher checkpoint found for dataset={dataset}")

print(best.as_posix())
PY
}

snapshot_student_names() {
  python3 - "$dataset" <<'PY'
from pathlib import Path
import re
import sys

dataset = sys.argv[1]
pattern = re.compile(rf"ssegcnbertshallow_{dataset}_acc_(\d+\.\d+)_f1_(\d+\.\d+)$")
for path in sorted(Path("state_dict").glob(f"ssegcnbertshallow_{dataset}_acc_*_f1_*")):
    if pattern.fullmatch(path.name):
        print(path.name)
PY
}

select_new_best_student() {
  python3 - "$dataset" "$1" <<'PY'
from pathlib import Path
import re
import sys

dataset = sys.argv[1]
before_path = Path(sys.argv[2])
pattern = re.compile(rf"ssegcnbertshallow_{dataset}_acc_(\d+\.\d+)_f1_(\d+\.\d+)$")
before = set(before_path.read_text().splitlines()) if before_path.exists() else set()

current = []
for path in Path("state_dict").glob(f"ssegcnbertshallow_{dataset}_acc_*_f1_*"):
    match = pattern.fullmatch(path.name)
    if match:
        metrics = tuple(float(x) for x in match.groups())
        current.append((metrics, path))

new_paths = [(metrics, path) for metrics, path in current if path.name not in before]
if not new_paths:
    raise SystemExit(f"No new student checkpoint found for dataset={dataset}")

best_metrics, best_path = max(new_paths, key=lambda item: item[0])
for _, path in new_paths:
    if path != best_path:
        path.unlink()

print(best_path.as_posix())
PY
}

run_stage() {
  local stage_name="$1"
  shift

  local before_file
  before_file="$(mktemp)"
  snapshot_student_names > "$before_file"

  echo "==> ${stage_name}"
  python3 ./train.py "$@"

  student_path="$(select_new_best_student "$before_file")"
  rm -f "$before_file"

  echo "==> ${stage_name} best checkpoint: ${student_path}"
}

teacher_path="$(find_best_teacher)"
student_path=""

common_args=(
  --model_name ssegcnbertshallow
  --dataset "$dataset"
  --seed 1000
  --num_epoch 2
  --batch_size 16
  --log_step 10
  --max_length 100
  --learning_rate 2e-4
  --teacher_path "$teacher_path"
  --teacher_model_name ssegcnbert
  --student_bert_layers 6
  --student_bert_use_adamw true
  --cuda 0
)

run_stage "stage54-uniform" \
  "${common_args[@]}" \
  --student_init_path "$teacher_path" \
  --student_bert_layer_map uniform \
  --bert_lr 2e-5 \
  --kd_temperature 2.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.5 \
  --kd_beta 0.35 \
  --kd_gamma 0.15 \
  --kd_feature_loss cosine \
  --kd_use_instance_weighting false

run_stage "stage55-last-logits" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 1.5e-5 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.55 \
  --kd_beta 0.45 \
  --kd_gamma 0.0 \
  --kd_feature_loss cosine \
  --kd_use_instance_weighting false

run_stage "stage56-tinybert-lite" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 1.2e-5 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.6 \
  --kd_beta 0.35 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.03 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.01 \
  --kd_use_instance_weighting false

run_stage "stage57-hidden-only" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 1e-5 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.6 \
  --kd_beta 0.35 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.04 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

run_stage "stage58-hidden-polish" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 8e-6 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.62 \
  --kd_beta 0.33 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.04 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

run_stage "stage59-accuracy-push" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 6e-6 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.66 \
  --kd_beta 0.29 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.03 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

run_stage "stage60-hardlabel-edge" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 5e-6 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.70 \
  --kd_beta 0.25 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.02 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

run_stage "stage61-hardlabel-polish" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 4e-6 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.72 \
  --kd_beta 0.23 \
  --kd_gamma 0.0 \
  --kd_token_hidden_weight 0.015 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

run_stage "stage64-dist-lastmile" \
  "${common_args[@]}" \
  --student_init_path "$student_path" \
  --student_bert_layer_map last \
  --bert_lr 3e-6 \
  --kd_temperature 3.0 \
  --kd_temperature_schedule constant \
  --kd_alpha 0.76 \
  --kd_beta 0.12 \
  --kd_gamma 0.0 \
  --kd_logit_mode dist \
  --kd_dist_inter_weight 1.0 \
  --kd_dist_intra_weight 2.0 \
  --kd_token_hidden_weight 0.01 \
  --kd_token_hidden_loss cosine \
  --kd_token_relation_weight 0.0 \
  --kd_use_instance_weighting false

echo "final_checkpoint=${student_path}"

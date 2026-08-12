#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="docker"
if [[ "${1:-}" == "--local" ]]; then
    MODE="local"
fi

run_config () {
    local CONFIG_PATH="$1"
    echo "=== ${CONFIG_PATH} ==="
    if [[ "$MODE" == "docker" ]]; then
        docker compose --profile gpu run --rm -t bci-gpu \
            python scripts/run_production.py --config "$CONFIG_PATH"
    else
        python scripts/run_production.py --config "$CONFIG_PATH"
    fi
    echo
}

# # Cho2017
# run_config "preprocessing/configs/Cho2017/Cho2017_s1_simple-CAR-128Hz-preproc.json"
# run_config "preprocessing/configs/Cho2017/Cho2017_s1_10ICA-CAR-128Hz-preproc.json"

# # Dreyer2023A
# run_config "preprocessing/configs/Dreyer2023A/Dreyer2023A_s1_simple-CAR-128Hz-preproc.json"
# run_config "preprocessing/configs/Dreyer2023A/Dreyer2023A_s1_10ICA-CAR-128Hz-preproc.json"

# # Lee2019
# run_config "preprocessing/configs/Lee2019/Lee2019_s1-2_simple-CAR-125Hz-preproc.json"
run_config "preprocessing/configs/Lee2019/Lee2019_s1-2_10ICA-CAR-125Hz-preproc.json"

echo "All done."

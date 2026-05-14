# Train MOFA+ on all three samples, then dump all factor loadings for each.

set -euo pipefail
cd "$(dirname "$0")"

samples=(
    "mouse_V11L12_038_D1:mofa_D1.py"
    "mouse_V11L12_038_B1:mofa_B1.py"
    "mouse_V11T16_085_C1:mofa_C1.py"
)

banner () {
    echo "  $1  ($(date '+%Y-%m-%d %H:%M:%S'))"
}

# Train MOFA on each sample 
for entry in "${samples[@]}"; do
    sample="${entry%%:*}"
    script="${entry##*:}"
    banner "Training MOFA on $sample  (script: $script)"
    python3 -u "$script"
done

# Dump all factor loadings for each sample 
for entry in "${samples[@]}"; do
    sample="${entry%%:*}"
    banner "Dumping all factors for $sample"
    python3 -u mofa_dump.py "$sample"
done

banner "ALL DONE!"
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${LEWORLD_ROOT_DIR:-$SCRIPT_ROOT}"
DATA_DIR="$ROOT_DIR/data"
TARGET_DIR="$DATA_DIR/leworld_input"
RAW_DIR="$DATA_DIR/raw"
INT_DIR="$DATA_DIR/integrated_processed"

mkdir -p "$DATA_DIR" "$RAW_DIR" "$INT_DIR"

link_candidate() {
  local source_dir="$1"
  local label="$2"
  local required=(
    "expression_matrix.npy"
    "metadata.csv"
    "gene_names.txt"
    "data_summary.json"
  )

  for name in "${required[@]}"; do
    if [[ ! -f "$source_dir/$name" ]]; then
      echo "Candidate is incomplete: $source_dir/$name is missing." >&2
      return 1
    fi
  done

  if [[ -L "$TARGET_DIR" ]]; then
    local current_target
    current_target="$(readlink -f "$TARGET_DIR")"
    local resolved_source
    resolved_source="$(readlink -f "$source_dir")"
    if [[ "$current_target" != "$resolved_source" ]]; then
      echo "Refusing to retarget existing data symlink: $TARGET_DIR -> $current_target" >&2
      echo "Candidate requested: $resolved_source" >&2
      return 1
    fi
    echo "Existing $label link already points to the verified candidate."
    return 0
  fi

  if [[ ! -e "$TARGET_DIR" ]]; then
    ln -s "$source_dir" "$TARGET_DIR"
    echo "Linked $label data: $TARGET_DIR -> $source_dir"
    return 0
  fi

  if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Refusing to replace non-directory target: $TARGET_DIR" >&2
    return 1
  fi

  local missing=()
  for name in "${required[@]}"; do
    local source_file="$source_dir/$name"
    local target_file="$TARGET_DIR/$name"
    if [[ -e "$target_file" || -L "$target_file" ]]; then
      if ! cmp -s "$source_file" "$target_file"; then
        echo "Refusing to overwrite conflicting file: $target_file" >&2
        return 1
      fi
    else
      missing+=("$name")
    fi
  done
  for name in "${missing[@]}"; do
    local source_file="$source_dir/$name"
    local target_file="$TARGET_DIR/$name"
    ln -s "$source_file" "$target_file"
    echo "Linked missing file: $target_file -> $source_file"
  done
  echo "Verified and completed existing data directory from $label source."
}

LOCAL_CANDIDATE="$INT_DIR/leworld_input"
if [[ -d "$LOCAL_CANDIDATE" ]]; then
  link_candidate "$LOCAL_CANDIDATE" "local integrated"
  exit $?
fi

PARENT_CANDIDATE="$ROOT_DIR/../output/integrated_processed/leworld_input"
if [[ -d "$PARENT_CANDIDATE" ]]; then
  link_candidate "$PARENT_CANDIDATE" "parent integrated"
  exit $?
fi

echo "No complete leworld_input candidate found." >&2
echo "Run preprocess first or place data at: $LOCAL_CANDIDATE" >&2
exit 1

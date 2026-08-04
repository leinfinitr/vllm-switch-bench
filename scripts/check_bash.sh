#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

found=0
while IFS= read -r -d '' script; do
  found=1
  bash -n "$script"
done < <(find scripts -maxdepth 1 -type f -name '*.sh' -print0 | sort -z)

if ((found == 0)); then
  printf 'no shell scripts found\n' >&2
  exit 1
fi
printf 'bash syntax: ok\n'

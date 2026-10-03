#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
checkout_dir="$repo_dir/baselines/Ayo"
patch_file="$repo_dir/patches/ayo-full-compat.patch"
commit=3748fc4482fcf8849a141aaaf0948eaf59a097d3
if [ ! -e "$checkout_dir" ]; then
    mkdir -p "$repo_dir/baselines"
    git -C "$repo_dir" worktree add --detach "$checkout_dir" "$commit"
fi
if [ "$(git -C "$checkout_dir" rev-parse HEAD)" != "$commit" ]; then
    echo "Unexpected upstream checkout revision: $checkout_dir" >&2
    exit 1
fi
if git -C "$checkout_dir" diff HEAD -- Ayo | cmp -s - "$patch_file"; then
    echo "Pinned Ayo compatibility checkout is ready: $checkout_dir"
    exit 0
fi
if [ -n "$(git -C "$checkout_dir" status --porcelain)" ]; then
    echo "Preserving existing checkout changes; inspect $checkout_dir" >&2
    exit 1
fi
git -C "$checkout_dir" apply --check "$patch_file"
git -C "$checkout_dir" apply "$patch_file"
echo "Prepared pinned Ayo compatibility checkout: $checkout_dir"

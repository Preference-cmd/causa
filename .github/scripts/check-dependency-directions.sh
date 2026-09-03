#!/usr/bin/env bash
# Dependency-direction guard (slice 11, first-principles item 7.1).
#
# Asserts the publish set's layering on the *normal* dependency graph:
#   causa-kernel  <- causa-protocol <- (causa-runtime, causa-provider)
#   causa-kernel  <- causa-extension
#   (causa-*, causa-extension) <- causa (facade; the only crate allowed
#   to depend on every family member, depended on by none)
# The kernel (facts + contracts) must never grow transport or policy
# baggage; the runtime (reference driver) must never depend on the edge
# adapters it drives. Dev-dependencies are exempt by design (tests and
# examples may wire the layers together; the published graph may not).
set -euo pipefail

check() {
  local root="$1"
  shift
  local tree
  tree=$(cargo tree -p "$root" --edges normal --charset ascii)
  local banned
  for banned in "$@"; do
    if grep -q -- "$banned" <<<"$tree"; then
      echo "FAIL: $root must not depend on $banned" >&2
      exit 1
    fi
  done
  echo "ok: $root"
}

# The kernel is facts + contracts only — no transport, no MCP, no
# subscriber, no sibling crates. ("causa v" is the facade's cargo-tree
# rendering; a bare "causa" would false-positive on the crate's own name.)
check causa-kernel reqwest rmcp axum tracing-subscriber \
  causa-protocol causa-runtime causa-provider causa-extension "causa v"

# Protocol translation stays pure: no transport, no driver.
check causa-protocol reqwest rmcp axum tracing-subscriber \
  causa-runtime causa-provider causa-extension "causa v"

# The reference driver drives the kernel and nothing else.
check causa-runtime reqwest rmcp axum tracing-subscriber \
  causa-provider causa-extension "causa v"

# The provider is the transport adapter — MCP is not its business.
check causa-provider rmcp axum causa-runtime causa-extension "causa v"

# Extensions are DynamicToolSource adapters — no driver, no protocol
# rendering, no server-side HTTP. (reqwest IS allowed here: rmcp's
# streamable-http *client* transport pulls it in transitively.)
check causa-extension axum \
  causa-runtime causa-provider causa-protocol "causa v"

# The facade may depend on the whole family but nothing may depend on
# it — a dependent would close a publish cycle.
for member in causa-kernel causa-protocol causa-runtime causa-provider causa-extension; do
  tree=$(cargo tree -p "$member" --edges normal --charset ascii)
  if grep -q -- "causa v" <<<"$tree"; then
    echo "FAIL: $member must not depend on the causa facade" >&2
    exit 1
  fi
done
echo "ok: no family member depends on the causa facade"

echo "dependency directions OK"

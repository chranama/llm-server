# cli/commands/k8s.py
from __future__ import annotations

import argparse

from cli.errors import CLIError
from cli.util.proc import ensure_bins, run_bash
from cli.types import GlobalConfig  # type: ignore[attr-defined]


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("k8s", help="kind/kubectl orchestration helpers.")
    p.set_defaults(_handler=_handle)

    sp = p.add_subparsers(dest="k8s_cmd", required=True)

    sp.add_parser("kind-up", help="Create kind cluster if missing")
    sp.add_parser("kind-down", help="Delete kind cluster")
    sp.add_parser("kind-ingress-up", help="Install ingress-nginx into kind")
    sp.add_parser("kind-build-server", help="Build server image + load into kind")

    sp.add_parser("apply-local-generate-only", help="kubectl apply -k deploy/k8s/overlays/local-generate-only")
    sp.add_parser("delete-local-generate-only", help="kubectl delete -k deploy/k8s/overlays/local-generate-only")

    sp.add_parser("apply-prod-gpu-full", help="kubectl apply -k deploy/k8s/overlays/prod-gpu-full")
    sp.add_parser("delete-prod-gpu-full", help="kubectl delete -k deploy/k8s/overlays/prod-gpu-full")

    sp.add_parser("wait", help="Wait for api deployment rollout + db-migrate job")
    sp.add_parser("status", help="kubectl get all/pods")
    sp.add_parser("logs-api", help="kubectl logs deployment/api -f")


def _handle(cfg: GlobalConfig, args: argparse.Namespace) -> int:
    ensure_bins("bash", "kubectl", "kind", "docker")

    K8S_DIR = cfg.repo_root / "deploy" / "k8s"
    KIND_CFG = K8S_DIR / "kind" / "kind-config.yaml"
    KIND_CLUSTER = "llm"
    NS = "llm"

    overlay_local = K8S_DIR / "overlays" / "local-generate-only"
    overlay_prod = K8S_DIR / "overlays" / "prod-gpu-full"

    c = args.k8s_cmd

    if c == "kind-up":
        run_bash(
            f'set -euo pipefail; '
            f'kind get clusters | grep -qx "{KIND_CLUSTER}" || kind create cluster --config "{KIND_CFG}"; '
            f'kubectl cluster-info >/dev/null; '
            f'echo "✅ kind cluster up: {KIND_CLUSTER}"',
            verbose=args.verbose,
        )
        return 0

    if c == "kind-down":
        run_bash(
            f'set -euo pipefail; '
            f'kind delete cluster --name "{KIND_CLUSTER}" || true; '
            f'echo "✅ kind cluster down: {KIND_CLUSTER}"',
            verbose=args.verbose,
        )
        return 0

    if c == "kind-ingress-up":
        run_bash(
            'set -euo pipefail; '
            'kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.0/deploy/static/provider/kind/deploy.yaml; '
            'kubectl -n ingress-nginx rollout status deployment/ingress-nginx-controller --timeout=180s; '
            'echo "✅ ingress-nginx installed"',
            verbose=args.verbose,
        )
        return 0

    if c == "kind-build-server":
        dockerfile = cfg.repo_root / "deploy" / "docker" / "Dockerfile.server"
        run_bash(
            f'set -euo pipefail; '
            f'docker build -t llm-server:dev -f "{dockerfile}" "{cfg.repo_root}"; '
            f'kind load docker-image llm-server:dev --name "{KIND_CLUSTER}"; '
            f'echo "✅ loaded llm-server:dev into kind"',
            verbose=args.verbose,
        )
        return 0

    if c == "apply-local-generate-only":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl apply -k "{overlay_local}"; '
            f'echo "✅ applied overlay: local-generate-only"',
            verbose=args.verbose,
        )
        return 0

    if c == "delete-local-generate-only":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl delete -k "{overlay_local}" --ignore-not-found; '
            f'echo "✅ deleted overlay: local-generate-only"',
            verbose=args.verbose,
        )
        return 0

    if c == "apply-prod-gpu-full":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl apply -k "{overlay_prod}"; '
            f'echo "✅ applied overlay: prod-gpu-full"',
            verbose=args.verbose,
        )
        return 0

    if c == "delete-prod-gpu-full":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl delete -k "{overlay_prod}" --ignore-not-found; '
            f'echo "✅ deleted overlay: prod-gpu-full"',
            verbose=args.verbose,
        )
        return 0

    if c == "wait":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl -n "{NS}" rollout status deployment/api --timeout=240s; '
            f'if kubectl -n "{NS}" get job db-migrate >/dev/null 2>&1; then '
            f'  echo "Waiting for job/db-migrate completion..."; '
            f'  kubectl -n "{NS}" wait --for=condition=complete job/db-migrate --timeout=240s || true; '
            f'  if ! kubectl -n "{NS}" get job db-migrate -o jsonpath="{{{{.status.conditions[?(@.type==\\"Complete\\")].status}}}}" 2>/dev/null | grep -q True; then '
            f'    echo "⚠️ job/db-migrate not complete (yet). Debug:"; '
            f'    kubectl -n "{NS}" describe job db-migrate || true; '
            f'    kubectl -n "{NS}" logs job/db-migrate --tail=200 || true; '
            f'  fi; '
            f'else '
            f'  echo "ℹ️ job/db-migrate not found (TTL may have cleaned it)."; '
            f'fi; '
            f'kubectl -n "{NS}" get pods; '
            f'echo "✅ k8s wait done"',
            verbose=args.verbose,
        )
        return 0

    if c == "status":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl -n "{NS}" get all; '
            f'kubectl -n "{NS}" get pods -o wide',
            verbose=args.verbose,
        )
        return 0

    if c == "logs-api":
        run_bash(
            f'set -euo pipefail; '
            f'kubectl -n "{NS}" logs deployment/api --tail=200 -f',
            verbose=args.verbose,
        )
        return 0

    raise CLIError(f"Unknown k8s command: {c}", code=2)
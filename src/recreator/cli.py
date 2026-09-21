"""Command line interface.

``recreator plan`` sizes a recreation before it starts, ``recreator graft`` builds the student and
verifies it against its donor, and ``recreator recreate`` does the whole pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("donor", help="Hugging Face repo id or local checkpoint directory.")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=("faithful", "balanced", "long_context"),
        help="HELIX hyper-parameters. 'faithful' stays closest to the donor. Default: balanced.",
    )
    parser.add_argument("--max-position", type=int, default=1 << 20, help="Context the student expects.")


def _cmd_plan(args: argparse.Namespace) -> int:
    from .donor import resolve_donor
    from .planner import plan_memory
    from .student import derive_student_config

    handle = resolve_donor(args.donor)
    config = derive_student_config(handle.config, args.preset, max_position_embeddings=args.max_position)
    teacher_parameters = args.teacher_parameters or _guess_donor_parameters(handle)
    plan = plan_memory(
        config,
        teacher_parameters=teacher_parameters,
        budget_gib=args.budget,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eight_bit_optimizer=args.eight_bit_optimizer,
        teacher_cached=args.cache_teacher,
    )
    print(f"donor       {args.donor} ({handle.architecture}{', MoE' if handle.is_moe else ''})")
    print(plan.render())
    return 0 if plan.fits else 1


def _guess_donor_parameters(handle) -> int:
    """Parameter count read from the donor's own tensors, so the plan is not guessing."""
    from .donor import iter_donor_tensors

    return sum(tensor.numel() for _, tensor in iter_donor_tensors(handle))


def _cmd_graft(args: argparse.Namespace) -> int:
    import torch

    from .graft import graft

    result = graft(
        args.donor,
        preset=args.preset,
        max_position_embeddings=args.max_position,
        identity_start=not args.no_identity_start,
        merge_mode=args.merge_mode,
        mlp_mode=args.mlp_mode,
        dtype=torch.float32,
    )
    print(result.report.summary())
    if result.sparse_layers:
        print(f"kept the donor sparse: {len(result.sparse_layers)} layers of experts and routers")
    if result.report.missing:
        print(f"left at init: {len(result.report.missing)} tensors (the new strands)")
    adapted = [entry for entry in result.report.transferred if entry.adapted]
    for entry in adapted[:10]:
        print(f"  adapted {entry.name}: {entry.method} {entry.notes}")

    if args.verify:
        from .verify import verify_against_donor

        outcome = verify_against_donor(result, args.donor)
        print(outcome.render())
        if not outcome.passed:
            return 1

    if args.output:
        from .export import save_recreated

        path = save_recreated(
            result.model,
            args.output,
            donor_path=result.donor_id,
            donor_id=args.donor,
            report=result.report,
            repo=Path(args.output).name,
        )
        print(f"written to {path}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .graft import graft
    from .verify import verify_against_donor

    result = graft(args.donor, preset=args.preset, max_position_embeddings=args.max_position)
    outcome = verify_against_donor(result, args.donor, seq_len=args.seq_len, tolerance=args.tolerance)
    print(outcome.render())
    return 0 if outcome.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recreator",
        description="Rebuild a pretrained transformer as a HELIX model without pretraining it again.",
    )
    parser.add_argument("--version", action="version", version=f"recreator {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Estimate whether a recreation fits a VRAM budget.")
    _add_shared(plan)
    plan.add_argument("--budget", type=float, default=96.0, help="VRAM budget in GiB. Default: 96.")
    plan.add_argument("--seq-len", type=int, default=4096)
    plan.add_argument("--batch-size", type=int, default=1)
    plan.add_argument("--teacher-parameters", type=int, default=0, help="Override the donor's counted size.")
    plan.add_argument("--eight-bit-optimizer", action="store_true")
    plan.add_argument("--cache-teacher", action="store_true", help="Plan with teacher outputs precomputed.")
    plan.set_defaults(func=_cmd_plan)

    build = sub.add_parser("graft", help="Build a HELIX student from a donor and report coverage.")
    _add_shared(build)
    build.add_argument("--output", help="Directory to write the grafted model to.")
    build.add_argument("--verify", action="store_true", help="Check the graft reproduces the donor.")
    build.add_argument("--no-identity-start", action="store_true", help="Skip silencing the new strands.")
    build.add_argument("--merge-mode", default="concat", choices=("concat", "average"))
    build.add_argument(
        "--mlp-mode",
        default="dense",
        choices=("dense", "sparse"),
        help="For a MoE donor: 'sparse' keeps every expert and the router, swapping only the "
        "attention. 'dense' collapses the experts, which is much smaller. Default: dense.",
    )
    build.set_defaults(func=_cmd_graft)

    check = sub.add_parser("verify", help="Check a graft reproduces its donor inside the local window.")
    _add_shared(check)
    check.add_argument("--seq-len", type=int, default=0, help="0 picks a length inside the local window.")
    check.add_argument("--tolerance", type=float, default=1e-3)
    check.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # surface a clean message rather than a traceback
        print(f"recreator: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

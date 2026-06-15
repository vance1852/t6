from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .data_generator import generate_problem
from .engine import SchedulingEngine
from .models import ScheduleProblem, ScheduleResult
from .validator import validate_schedule
from .reporter import (
    print_terminal_report,
    export_csv,
    export_json,
)
from .comparator import compare_strategies, print_comparison_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="charge-sched",
        description="小区集中充电智能排程工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")

    p_gen = sub.add_parser("generate", help="生成模拟充电数据")
    p_gen.add_argument("--ev", type=int, default=15, help="电动汽车数量（默认15）")
    p_gen.add_argument("--ebike", type=int, default=40, help="电动自行车数量（默认40）")
    p_gen.add_argument("--seed", type=int, default=42, help="随机种子（默认42，可复现）")
    p_gen.add_argument("--overload", type=float, default=1.0, help="过载系数，>1时故意制造供不应求场景")
    p_gen.add_argument("-o", "--output", type=str, default="problem.json", help="输出数据文件路径")

    p_run = sub.add_parser("run", help="运行指定策略生成排程")
    p_run.add_argument("-i", "--input", type=str, required=True, help="问题数据文件")
    p_run.add_argument("-s", "--strategy", type=str, required=True,
                        choices=["fcfs", "edf", "valley_fill", "optimized"],
                        help="调度策略")
    p_run.add_argument("-o", "--output", type=str, default=None,
                       help="排程结果输出目录（不指定则仅打印）")
    p_run.add_argument("--no-report", action="store_true", help="不打印终端报表")
    p_run.add_argument("--no-validate", action="store_true", help="不自动校验排程")

    p_report = sub.add_parser("report", help="从排程结果打印报表和导出")
    p_report.add_argument("-p", "--problem", type=str, required=True, help="问题数据文件")
    p_report.add_argument("-r", "--result", type=str, required=True, help="排程结果文件")
    p_report.add_argument("-o", "--output", type=str, default=None, help="导出目录")
    p_report.add_argument("--no-terminal", action="store_true", help="不打印终端报表")

    p_val = sub.add_parser("validate", help="校验排程结果合法性")
    p_val.add_argument("-p", "--problem", type=str, required=True, help="问题数据文件")
    p_val.add_argument("-r", "--result", type=str, required=True, help="排程结果文件")

    p_cmp = sub.add_parser("compare", help="并排对比多个策略")
    p_cmp.add_argument("-i", "--input", type=str, required=True, help="问题数据文件")
    p_cmp.add_argument("-s", "--strategies", nargs="+",
                        default=["fcfs", "edf", "valley_fill", "optimized"],
                        help="要对比的策略列表（默认全部四种）")
    p_cmp.add_argument("-o", "--output", type=str, default=None, help="结果导出目录")

    return parser


def cmd_generate(args) -> int:
    problem = generate_problem(
        num_ev=args.ev,
        num_ebike=args.ebike,
        seed=args.seed,
        overload_factor=args.overload,
    )
    problem.save(args.output)
    total_ev = sum(1 for s in problem.sessions if s.vehicle_type.value == "ev")
    total_eb = len(problem.sessions) - total_ev
    print(f"已生成模拟数据并保存至: {args.output}")
    print(f"  电动汽车: {total_ev} 辆")
    print(f"  电动自行车: {total_eb} 辆")
    print(f"  总充电会话: {len(problem.sessions)} 条")
    print(f"  时间片数: {problem.num_slots} (15分钟/片)")
    if args.overload > 1.0:
        print(f"  过载系数: {args.overload:.2f}")
    return 0


def cmd_run(args) -> int:
    problem = ScheduleProblem.load(args.input)
    engine = SchedulingEngine()
    result = engine.run(problem, args.strategy)

    if not args.no_validate:
        report = validate_schedule(result)
        print(report.summary())
        for v in report.violations:
            print(f"  {v}")

    if not args.no_report:
        print_terminal_report(result)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        csv_paths = export_csv(result, args.output)
        json_path = export_json(result, args.output)
        print()
        print(f"结果已导出:")
        for k, v in csv_paths.items():
            print(f"  {k}: {v}")
        print(f"  json: {json_path}")

    return 0


def cmd_report(args) -> int:
    problem = ScheduleProblem.load(args.problem)
    result = ScheduleResult.load(args.result, problem)
    if not args.no_terminal:
        print_terminal_report(result)
    if args.output:
        csv_paths = export_csv(result, args.output)
        json_path = export_json(result, args.output)
        print()
        print(f"结果已导出:")
        for k, v in csv_paths.items():
            print(f"  {k}: {v}")
        print(f"  json: {json_path}")
    return 0


def cmd_validate(args) -> int:
    problem = ScheduleProblem.load(args.problem)
    result = ScheduleResult.load(args.result, problem)
    report = validate_schedule(result)
    print(report.summary())
    print()
    if report.violations:
        for v in report.violations:
            print(f"  {v}")
    else:
        print("  未发现任何违规项。")
    return 0 if report.is_valid else 1


def cmd_compare(args) -> int:
    problem = ScheduleProblem.load(args.input)
    results = compare_strategies(problem, args.strategies)
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        for name, result in results.items():
            export_csv(result, args.output)
            export_json(result, args.output)
        print()
        print(f"各策略结果已导出至: {args.output}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "generate": cmd_generate,
        "run": cmd_run,
        "report": cmd_report,
        "validate": cmd_validate,
        "compare": cmd_compare,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

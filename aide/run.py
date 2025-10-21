import atexit
import logging
import os
import shutil

from . import backend

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from typing import cast
import sys

logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def print_plain_log(cfg, journal, global_step, task_desc_str, status_text):
    """Print plain text log output instead of Rich Live display"""
    print(f"\n{'='*80}")
    print(f"AIDE is working on experiment: {cfg.exp_name}")
    print(f"{'='*80}")
    
    # Task description
    print(f"\nTask Description:")
    print(f"{'-'*40}")
    print(task_desc_str.strip())
    
    # Progress
    progress_percent = (global_step / cfg.agent.steps) * 100
    print(f"\nProgress: {global_step}/{cfg.agent.steps} ({progress_percent:.1f}%)")
    print(f"Status: {status_text}")
    
    # Solution tree
    print(f"\nSolution Tree:")
    print(f"{'-'*40}")
    best_node = journal.get_best_node()
    
    def print_node(node: Node, indent=0):
        prefix = "  " * indent
        if node.is_buggy:
            print(f"{prefix}◍ bug")
        else:
            if node is best_node:
                print(f"{prefix}● {node.metric.value:.3f} (best)")
            else:
                print(f"{prefix}● {node.metric.value:.3f}")
        
        for child in node.children:
            print_node(child, indent + 1)
    
    for n in journal.draft_nodes:
        print_node(n)
    
    # File paths
    print(f"\nFile Paths:")
    print(f"{'-'*40}")
    print(f"Result visualization: {cfg.log_dir / 'tree_plot.html'}")
    print(f"Agent workspace directory: {cfg.workspace_dir}")
    print(f"Experiment log directory: {cfg.log_dir}")
    
    print(f"\nPress Ctrl+C to stop the run")
    print(f"{'='*80}\n")


def run():
    cfg = load_cfg()
    logger.info(f'Starting run "{cfg.exp_name}"')

    logger.info(f'Config.log_dir: {cfg.log_dir}')
    logger.info(f'Config.workspace_dir: {cfg.workspace_dir}')

    # Check environment variable for plain log mode
    use_plain_log = os.getenv('AIDE_PLAIN_LOG', 'false').lower() in ('true', '1', 'yes')
    
    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    if use_plain_log:
        print("Preparing agent workspace (copying and extracting files) ...")
        prep_agent_workspace(cfg)
    else:
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    global_step = len(journal)
    
    if use_plain_log:
        # Plain log mode - no Rich objects needed
        prog = None
        status = None
    else:
        # Rich Live display mode - create Rich objects
        prog = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        status = Status("[green]Generating code...")
        prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        if use_plain_log:
            print("Status: Executing code...")
        else:
            if status is not None:
                status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        if use_plain_log:
            print("Status: Generating code...")
        else:
            if status is not None:
                status.update("[green]Generating code...")
        return res

    def generate_live():
        tree = journal_to_rich_tree(journal)
        if prog is not None:
            prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        
        # Only create Group with prog and status if they are not None
        if prog is not None and status is not None:
            left = Group(
                Panel(Text(task_desc_str.strip()), title="Task description"), prog, status
            )
        else:
            left = Panel(Text(task_desc_str.strip()), title="Task description")
            
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    if use_plain_log:
        # Plain log mode - print updates without Live display
        print_plain_log(cfg, journal, global_step, task_desc_str, "Generating code...")
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=exec_callback)
            save_run(cfg, journal)
            global_step = len(journal)
            print_plain_log(cfg, journal, global_step, task_desc_str, "Generating code...")
    else:
        # Rich Live display mode (original behavior)
        with Live(
            generate_live(),
            refresh_per_second=16,
            screen=True,
        ) as live:
            while global_step < cfg.agent.steps:
                agent.step(exec_callback=exec_callback)
                save_run(cfg, journal)
                global_step = len(journal)
                live.update(generate_live())
    interpreter.cleanup_session()

    if cfg.generate_report:
        print("Generating final report from journal...")
        report, llm_info = journal2report(journal, task_desc, cfg.report)
        report = cast(str, report)
        print("\n\nReport:\n", report)
        print("\n\nLLM info:\n", llm_info)
        llm_info = '# LLM INFO:\n' + str(llm_info)
        report = report + '\n\n' + llm_info
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()

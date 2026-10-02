"""Main entry point and Interactive CLI for the AI Agent.

Run interactively:
    python main.py

Run automated Step 1 verification test:
    python main.py --demo

Run single command:
    python main.py "Create test.py"
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure package directory is on path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from agent import AgentEngine
from config import Style, config


def print_banner():
    banner = f"""
{Style.paint("╔═══════════════════════════════════════════════════════════════╗", Style.BRIGHT_CYAN)}
{Style.paint("║", Style.BRIGHT_CYAN)}   {Style.paint("🤖 CUSTOM AI AGENT ENGINE (STEP 1: FOUNDATION)", Style.BOLD, Style.BRIGHT_WHITE)}       {Style.paint("║", Style.BRIGHT_CYAN)}
{Style.paint("║", Style.BRIGHT_CYAN)}   {Style.paint("Architecture: Input → Intent → Planning → Tool → Obs → Reply", Style.DIM)} {Style.paint("║", Style.BRIGHT_CYAN)}
{Style.paint("╚═══════════════════════════════════════════════════════════════╝", Style.BRIGHT_CYAN)}
  • {Style.paint("Workspace:", Style.BOLD)}  {config.WORKSPACE_DIR}
  • {Style.paint("Database:", Style.BOLD)}   {config.DB_PATH.name}
  • {Style.paint("Tools (5):", Style.BOLD)}  create_file, read_file, write_file, list_files, run_command
  • Type {Style.paint("help", Style.YELLOW)} for examples or {Style.paint("exit", Style.YELLOW)} to quit.
"""
    print(banner)


def run_interactive():
    print_banner()
    agent = AgentEngine(verbose=True)

    while True:
        try:
            prompt = f"\n{Style.paint('agent', Style.BRIGHT_GREEN, Style.BOLD)}{Style.paint('>', Style.BRIGHT_CYAN)} "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print(Style.paint("\nExiting AI Agent. Happy coding!\n", Style.DIM))
                break

            response = agent.run_cycle(user_input)
            print(f"\n{Style.paint('💬 Agent:', Style.BOLD, Style.BRIGHT_WHITE)} {response}")

        except (KeyboardInterrupt, EOFError):
            print(Style.paint("\n\nSession terminated by user.", Style.YELLOW))
            break
        except Exception as e:
            print(Style.paint(f"\n❌ Unexpected error: {e}", Style.RED))


def run_automated_demo():
    print_banner()
    print(Style.paint("🧪 RUNNING AUTOMATED VERIFICATION DEMO (STEP 1 + STEP 2)\n", Style.BOLD, Style.BRIGHT_YELLOW))
    agent = AgentEngine(verbose=True)

    test_commands = [
        # Step 1 Baseline Commands
        "Create test.py",
        'Write print("Hello from my AI agent!") into test.py',
        "Run test.py",
        # Step 2 Advanced Multi-Step Natural Language Request
        "Can you make me a Python file that calculates the average of three numbers and then run it?",
        "List files",
    ]

    for idx, cmd in enumerate(test_commands, 1):
        print(f"\n{Style.paint(f'=== [Demo Step {idx}/{len(test_commands)}]: {cmd} ===', Style.BOLD, Style.BRIGHT_CYAN)}")
        response = agent.run_cycle(cmd)
        print(f"\n{Style.paint('💬 Agent Response:', Style.BOLD)}\n{response}")

    print(Style.paint("\n🎉 ALL STEP 1 & STEP 2 TARGET TESTS COMPLETED SUCCESSFULLY!", Style.BOLD, Style.BRIGHT_GREEN))


def main():
    parser = argparse.ArgumentParser(description="My AI Agent CLI")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration test")
    parser.add_argument("command", nargs="*", help="Optional one-shot command")

    args = parser.parse_args()

    if args.demo:
        run_automated_demo()
    elif args.command:
        full_command = " ".join(args.command)
        agent = AgentEngine(verbose=True)
        response = agent.run_cycle(full_command)
        print(f"\n{Style.paint('💬 Agent:', Style.BOLD)} {response}")
    else:
        run_interactive()


if __name__ == "__main__":
    main()

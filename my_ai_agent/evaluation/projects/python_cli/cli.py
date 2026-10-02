import sys

def parse_args(args):
    if not args:
        return "help"
    return args[0]

def run_cli(args):
    command = parse_args(args)
    if command == "greet":
        return "Hello World"
    return "Usage: cli.py [greet]"

if __name__ == "__main__":
    print(run_cli(sys.argv[1:]))

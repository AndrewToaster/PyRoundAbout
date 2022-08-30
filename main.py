if __name__ == '__main__':
    from runtime import State, step_state
    from utilities import InputThread, LockedListIO, GetchIO
    from sys import stdout, stdin as _stdin
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs='?', help="specifies the file which to execute")
    parser.add_argument("--input", help="Changes the way input is gathered. "
                                        "Raw uses program's STDIN, "
                                        "Getch: Use unbuffered STDIN, "
                                        "Proxy: Like raw, uses program's STDIN, but input is gathered off-thread",
                        choices=["raw", "getch", "proxy"], default="proxy")
    parser.add_argument("--width", type=int, help="overrides grid width")
    parser.add_argument("--height", type=int, help="overrides grid height")

    result = parser.parse_args()

    stdin = None
    if result.input == "proxy":
        stdin = LockedListIO()
        inp_thread = InputThread(lambda x: stdin.write(x))
        inp_thread.start()
    elif result.input == "raw":
        stdin = _stdin
    elif result.input == "getch":
        stdin = GetchIO()

    state = None
    with open(result.file, 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
        w = 0
        h = 0
        if len(content) != 0:
            line = content[0]
            if line.startswith('//'):
                parts = line[2:].split(',')
                w = int(parts[0])
                h = int(parts[1])
                content = content[1:]
            else:
                w = max(map(lambda x: len(x), content))
                h = len(content)
            state = State.create('\n'.join(content), w, h, stdin, stdout)
        else:
            raise Exception

    while True:
        if not step_state(state):
            break

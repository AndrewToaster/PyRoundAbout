from os import name


if name == 'nt':
    import msvcrt

    def getch():
        return msvcrt.getwch()

    # noinspection SpellCheckingInspection
    def kbhit():
        return msvcrt.kbhit()
elif name == 'linux':
    import sys
    import tty
    import termios
    import select

    # noinspection SpellCheckingInspection
    def kbhit():
        (x, _, _) = select.select([sys.stdin], [], [], 0)
        return len(x) != 0


    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)

        ch = sys.stdin.read(1)

        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
else:
    def getch():
        raise NotImplementedError

    # noinspection SpellCheckingInspection
    def kbhit():
        raise NotImplementedError

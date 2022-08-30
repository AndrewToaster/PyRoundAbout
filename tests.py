from __future__ import annotations
import unittest
from io import StringIO
from functools import wraps
import runtime as rt
from utilities import parse_map
from random import randint


def step_state(state: rt.State, steps: int) -> (bool, int):
    for i in range(steps):
        if not rt.step_state(state):
            return i == steps - 1, i + 1
    return True, steps


def setup(content: str, __input: str = None, width: int = None, height: int = None):
    def decorator(f):
        # noinspection PyFinal
        @wraps(f)
        def wrapper(self: StateTestCase):
            self.state.reset()
            self.stdin = StringIO(__input)
            self.state.stdin = self.stdin
            self.state.stdin_buffer.handle = self.stdin
            self.stdout = StringIO()
            self.state.stdout = self.stdout
            self.completed = None
            self.halted = None
            # PyCharm kept complaining
            w = width
            h = height
            if w is None or h is None:
                lines = content.splitlines()
                h = h or len(lines)
                w = w or len(max(lines))
            self.state.w = w
            self.state.initial_w = w
            self.state.h = h
            self.state.initial_h = h
            self.state.map = parse_map(content, w, h)
            self.state.initial_map = self.state.map
            self.preTest()
            f(self)
        return wrapper
    return decorator


class StateTestCase(unittest.TestCase):
    # noinspection PyTypeChecker
    @classmethod
    def setUpClass(cls) -> None:
        cls.stdin: StringIO = StringIO()
        cls.stdout: StringIO = StringIO()
        cls.state = rt.State.create("", 1, 1, None, None)
        cls.init()

    def preTest(self):
        ...

    def postTest(self):
        ...

    @classmethod
    def init(cls):
        ...

    def tearDown(self) -> None:
        print(self.stdout.getvalue())

    def step(self) -> bool:
        self.halted = rt.step_state(self.state)
        return self.halted

    def assertHalted(self):
        self.assertTrue(self.halted, "The program didn't end by halting")

    def assertStackUnchanged(self):
        self.assertEqual(len(self.state.stack), 0, "Stack was modified")

    def assertStackChanged(self):
        self.assertNotEqual(len(self.state.stack), 0, "Stack was not modified")

    def assertStack(self, value: list[int]):
        self.assertListEqual(self.state.stack, value, "Stack was not equal to expected value")

    def assertPointerUnchanged(self):
        self.assertEqual(self.state.pointer, 0, "Pointer was modified")

    def assertPointChanged(self):
        self.assertNotEqual(self.state.pointer, 0, "Pointer was not modified")

    def assertPointer(self, value: int):
        self.assertEqual(self.state.pointer, value, "Pointer was not equal to expected value")

    def assertMapUnchanged(self):
        self.assertListEqual(self.state.map, self.state.initial_map, "Map was modified")

    def assertMapChanged(self):
        self.assertNotEqual(self.state.map, self.state.initial_map, "Map was not modified")

    def assertMap(self, value: list[list[str]]):
        self.assertListEqual(self.state.map, value, "Map was not equal to expected value")

    def assertHeapUnchanged(self):
        self.assertEqual(len(self.state.heap), 0, "Heap was modified")

    def assertHeapChanged(self):
        self.assertNotEqual(len(self.state.heap), 0, "Heap was not modified")

    def assertHeap(self, value: dict[int, int]):
        value = {key: value for key, value in value.items() if value != 0}
        self.assertDictEqual(self.state.heap, value)

    def assertXUnchanged(self):
        self.assertEqual(self.state.x, 0, "Cursor X was modified")

    def assertXChanged(self):
        self.assertNotEqual(self.state.x, 0, "Cursor X was not modified")

    def assertX(self, value: int):
        self.assertEqual(self.state.x, value, "Cursor X was not equal to expected value")

    def assertYUnchanged(self):
        self.assertEqual(self.state.y, 0, "Cursor Y was modified")

    def assertYChanged(self):
        self.assertNotEqual(self.state.y, 0, "Cursor Y was not modified")

    def assertY(self, value: int):
        self.assertEqual(self.state.y, value, "Cursor Y was not equal to expected value")

    def assertWUnchanged(self):
        self.assertEqual(self.state.w, self.state.initial_w, "Map width was modified")

    def assertWChanged(self):
        self.assertNotEqual(self.state.w, self.state.initial_w, "Map width was not modified")

    def assertW(self, value: int):
        self.assertEqual(self.state.w, value, "Map width was not equal to expected value")

    def assertHUnchanged(self):
        self.assertEqual(self.state.h, self.state.initial_h, "Map height was modified")

    def assertHChanged(self):
        self.assertNotEqual(self.state.h, self.state.initial_h, "Map height was not modified")

    def assertH(self, value: int):
        self.assertEqual(self.state.h, value, "Map height was not equal to expected value")

    def assertFlagTrue(self, flag: int):
        self.assertTrue((self.state.flags & flag) != 0, "Flag was not True")

    def assertFlagFalse(self, flag: int):
        self.assertTrue((self.state.flags & flag) == 0, "Flag was not False")

    def assertFlag(self, flag: int, expected: bool):
        if expected:
            self.assertFlagTrue(flag)
        else:
            self.assertFlagFalse(flag)

    def assertModeUnchanged(self):
        self.assertEqual(self.state.mode, rt.Mode.Traversal, "Mode was changed")

    def assertModeChanged(self):
        self.assertNotEqual(self.state.mode, rt.Mode.Traversal, "Mode was not changed")

    def assertMode(self, value: rt.Mode):
        self.assertEqual(self.state.mode, value, "Mode was not equal to expected value")

    def assertDirectionUnchanged(self):
        self.assertEqual(self.state.direction, rt.Direction.Right, "Direction was changed")

    def assertDirectionChanged(self):
        self.assertNotEqual(self.state.direction, rt.Direction.Right, "Direction was not changed")

    def assertDirection(self, value: rt.Direction):
        self.assertEqual(self.state.direction, value, "Direction was not equal to expected value")

    def assertSymbol(self, value: str):
        self.assertEqual(self.state.current_symbol, value, "Cursor was at the wrong symbol")


# ------------------------------------- #
# noinspection SpellCheckingInspection
class TraversalTestCase(StateTestCase):
    @setup("~ ")
    def test_halt(self):
        self.assertFalse(self.step(), "Program didn't end by halting")

    @setup(";")
    def test_reset(self):
        for mode in rt.Mode:
            with self.subTest(mode=mode):
                self.state.mode = mode
                self.step()
                self.assertMode(rt.Mode.Traversal)

    @setup("@?%=[$#")
    def test_set(self):
        modes = [rt.Mode.ConditionalTraversal, rt.Mode.Comparison, rt.Mode.Operation,
                 rt.Mode.Stack, rt.Mode.Heap, rt.Mode.IO, rt.Mode.Map]
        x = 0
        for mode in modes:
            with self.subTest(mode=mode, i=x):
                self.state.mode = rt.Mode.Traversal
                self.step()
                x = (x + 1) % len(modes)
                self.assertMode(mode)
                self.assertX(x)

    @setup("   \n   \n   ")
    def test_directions(self):
        data = [
            (rt.Direction.Right, 1, 0),
            (rt.Direction.Left, 2, 0),
            (rt.Direction.Down, 0, 1),
            (rt.Direction.Up, 0, 2),
            (rt.Direction.Right | rt.Direction.Down, 1, 1),
            (rt.Direction.Right | rt.Direction.Up, 1, 2),
            (rt.Direction.Left | rt.Direction.Down, 2, 1),
            (rt.Direction.Left | rt.Direction.Up, 2, 2)
        ]
        for direction, x, y in data:
            with self.subTest(direction=direction, x=x, y=y):
                self.state.x = 0
                self.state.y = 0
                self.state.direction = direction
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(direction)

    @setup("v<\n>^")
    def test_arrows_flow(self):
        data = [
            (rt.Direction.Down, 0, 1),
            (rt.Direction.Right, 1, 1),
            (rt.Direction.Up, 1, 0),
            (rt.Direction.Left, 0, 0)
        ]
        for direction, x, y in data:
            with self.subTest(direction=direction, x=x, y=y):
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(direction)

    @setup("   \n / \n   ")
    def test_fsreflector(self):
        data = [
            (rt.Direction.Right, rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Left, rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Up, rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Down, rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Right | rt.Direction.Down, rt.Direction.Left | rt.Direction.Up, 0, 0),
            (rt.Direction.Right | rt.Direction.Up, rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Left | rt.Direction.Down, rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Left | rt.Direction.Up, rt.Direction.Right | rt.Direction.Down, 2, 2),
        ]
        for tdir, edir, x, y in data:
            with self.subTest(tdir=tdir, edir=edir, x=x, y=y):
                self.state.x = 1
                self.state.y = 1
                self.state.direction = tdir
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(edir)

    @setup("   \n \\ \n   ")
    def test_bsreflector(self):
        data = [
            (rt.Direction.Right, rt.Direction.Right | rt.Direction.Down, 2, 2),
            (rt.Direction.Left, rt.Direction.Left | rt.Direction.Up, 0, 0),
            (rt.Direction.Up, rt.Direction.Left | rt.Direction.Up, 0, 0),
            (rt.Direction.Down, rt.Direction.Right | rt.Direction.Down, 2, 2),
            (rt.Direction.Right | rt.Direction.Down, rt.Direction.Right | rt.Direction.Down, 2, 2),
            (rt.Direction.Right | rt.Direction.Up, rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Left | rt.Direction.Down, rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Left | rt.Direction.Up, rt.Direction.Left | rt.Direction.Up, 0, 0),
        ]
        for tdir, edir, x, y in data:
            with self.subTest(tdir=tdir, edir=edir, x=x, y=y):
                self.state.x = 1
                self.state.y = 1
                self.state.direction = tdir
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(edir)

    @setup("   \n | \n   ")
    def test_hreflector(self):
        data = [
            (rt.Direction.Right, rt.Direction.Left, 0, 1),
            (rt.Direction.Left, rt.Direction.Right, 2, 1),
            (rt.Direction.Up, rt.Direction.Up, 1, 0),
            (rt.Direction.Down, rt.Direction.Down, 1, 2),
            (rt.Direction.Right | rt.Direction.Down, rt.Direction.Down, 1, 2),
            (rt.Direction.Right | rt.Direction.Up, rt.Direction.Up, 1, 0),
            (rt.Direction.Left | rt.Direction.Down, rt.Direction.Down, 1, 2),
            (rt.Direction.Left | rt.Direction.Up, rt.Direction.Up, 1, 0),
        ]
        for tdir, edir, x, y in data:
            with self.subTest(tdir=tdir, edir=edir, x=x, y=y):
                self.state.x = 1
                self.state.y = 1
                self.state.direction = tdir
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(edir)

    @setup("   \n - \n   ")
    def test_vreflector(self):
        data = [
            (rt.Direction.Right, rt.Direction.Right, 2, 1),
            (rt.Direction.Left, rt.Direction.Left, 0, 1),
            (rt.Direction.Up, rt.Direction.Down, 1, 2),
            (rt.Direction.Down, rt.Direction.Up, 1, 0),
            (rt.Direction.Right | rt.Direction.Down, rt.Direction.Right, 2, 1),
            (rt.Direction.Right | rt.Direction.Up, rt.Direction.Right, 2, 1),
            (rt.Direction.Left | rt.Direction.Down, rt.Direction.Left, 0, 1),
            (rt.Direction.Left | rt.Direction.Up, rt.Direction.Left, 0, 1),
        ]
        for tdir, edir, x, y in data:
            with self.subTest(tdir=tdir, edir=edir, x=x, y=y):
                self.state.x = 1
                self.state.y = 1
                self.state.direction = tdir
                self.step()
                self.assertX(x)
                self.assertY(y)
                self.assertDirection(edir)

    @setup("   \n + \n   ")
    def test_preflector(self):
        data = {
            (rt.Direction.Left, 0, 1),
            (rt.Direction.Right, 2, 1),
            (rt.Direction.Up, 1, 0),
            (rt.Direction.Down, 1, 2)
        }
        results = set()
        for _ in range(1000):
            self.state.x = 1
            self.state.y = 1
            self.state.direction = rt.Direction.Up
            self.step()

            result = (self.state.direction, self.state.x, self.state.y)
            self.assertIn(result, data, "Result was not expected")
            results.add(result)

            if results == data:
                return
        self.assertSetEqual(data, results)

    @setup("   \n x \n   ")
    def test_xreflector(self):
        data = {
            (rt.Direction.Left | rt.Direction.Up, 0, 0),
            (rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Right | rt.Direction.Down, 2, 2)
        }
        results = set()
        for _ in range(1000):
            self.state.x = 1
            self.state.y = 1
            self.state.direction = rt.Direction.Up
            self.step()

            result = (self.state.direction, self.state.x, self.state.y)
            self.assertIn(result, data, "Result was not expected")
            results.add(result)

            if results == data:
                return
        self.assertSetEqual(data, results)

    @setup("   \n * \n   ")
    def test_sreflector(self):
        data = {
            (rt.Direction.Left, 0, 1),
            (rt.Direction.Right, 2, 1),
            (rt.Direction.Up, 1, 0),
            (rt.Direction.Down, 1, 2),
            (rt.Direction.Left | rt.Direction.Up, 0, 0),
            (rt.Direction.Left | rt.Direction.Down, 0, 2),
            (rt.Direction.Right | rt.Direction.Up, 2, 0),
            (rt.Direction.Right | rt.Direction.Down, 2, 2)
        }
        results = set()
        for _ in range(1000):
            self.state.x = 1
            self.state.y = 1
            self.state.direction = rt.Direction.Up
            self.step()

            result = (self.state.direction, self.state.x, self.state.y)
            self.assertIn(result, data, "Result was not expected")
            results.add(result)

            if results == data:
                return
        self.assertSetEqual(data, results)

    @setup("v_\n~ ")
    def test_flowtest(self):
        with self.subTest(flag=True):
            self.state.mode = rt.Mode.ConditionalTraversal
            self.state.set_flag(rt.Flags.ResultFlag, True)
            self.step()

            self.assertX(0)
            self.assertY(1)
            self.assertSymbol('~')
            self.assertDirection(rt.Direction.Down)

        with self.subTest(flag=False):
            self.state.reset()
            self.state.mode = rt.Mode.ConditionalTraversal
            self.state.set_flag(rt.Flags.ResultFlag, False)
            self.step()

            self.assertX(1)
            self.assertY(0)
            self.assertSymbol('_')
            self.assertDirection(rt.Direction.Right)


class ComparisonTestCase(StateTestCase):
    @classmethod
    def init(cls) -> None:
        cls.data = [(1, 1), (1, 0), *[(randint(-100, 100), randint(-100, 100)) for _ in range(10)]]

    def preTest(self):
        self.state.mode = rt.Mode.Comparison

    @setup(">")
    def test_gt(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                self.state.stack.clear()

                self.assertFlag(rt.Flags.ResultFlag, a > b)

    @setup("<")
    def test_lt(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                self.state.stack.clear()
                self.assertFlag(rt.Flags.ResultFlag, a < b)

    @setup("=")
    def test_eq(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                self.state.stack.clear()
                self.assertFlag(rt.Flags.ResultFlag, a == b)

    @setup("!")
    def test_neq(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                self.state.stack.clear()
                self.assertFlag(rt.Flags.ResultFlag, a != b)


class FlagsTestCase(StateTestCase):
    def preTest(self):
        self.state.mode = rt.Mode.Flags

    @setup('|')
    def test_true(self):
        for bit in range(16):
            with self.subTest(bit=bit):
                flag = 1 << bit
                self.state.stack.append(flag)
                self.step()
                self.assertFlagTrue(flag)

    @setup('&')
    def test_false(self):
        self.state.flags = 2 ** 16 - 1
        for bit in range(16):
            with self.subTest(bit=bit):
                flag = 1 << bit
                self.state.stack.append(flag)
                self.step()
                self.assertFlagFalse(flag)

    @setup('^')
    def test_invert(self):
        for bit in range(16):
            flag = 1 << bit
            _f = self.state.flags
            self.state.stack.append(flag)
            self.state.stack.append(flag)
            with self.subTest(bit=bit):
                with self.subTest(expected=True):
                    self.step()
                    self.assertFlagTrue(flag)
                with self.subTest(expected=False):
                    self.step()
                    self.assertFlagFalse(flag)

    @setup('>')
    def test_push(self):
        flag = randint(0, 2 ** 16 - 1)
        self.state.flags = flag
        self.step()
        self.assertStack([flag])

    @setup('?')
    def test_test(self):
        with self.subTest(expected=True):
            flag = randint(0, 2 ** 16 - 1)
            self.state.flags = flag
            self.state.stack.append(flag)
            self.step()
            self.assertFlagTrue(rt.Flags.ResultFlag)
        with self.subTest(expected=False):
            flag = randint(1, 2 ** 16 - 1)
            self.state.flags = flag - 1
            self.state.stack.append(flag)
            self.step()
            self.assertFlagFalse(rt.Flags.ResultFlag)


class OperationTestCase(StateTestCase):
    @classmethod
    def init(cls) -> None:
        cls.data = [(1, 0), (0, -1), *[(randint(-100, 100), randint(-100, 100)) for _ in range(10)]]

    def preTest(self):
        self.state.mode = rt.Mode.Operation

    @setup("+")
    def test_add(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a + b)

    @setup("-")
    def test_sub(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a - b)

    @setup("*")
    def test_mul(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a * b)

    @setup("/")
    def test_div(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if b != 0:
                    result, rem = divmod(a, b)
                    self.assertEqual(val, result)
                    self.assertFlag(rt.Flags.ResultTruncated, rem != 0)
                    self.assertFlagFalse(rt.Flags.DivisionByZero)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.DivisionByZero)
                    self.assertFlagFalse(rt.Flags.ResultTruncated)

    @setup("^")
    def test_pow(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if a != 0 or b > 0:
                    self.assertEqual(val, int(a ** b))
                    self.assertFlagFalse(rt.Flags.DivisionByZero)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.DivisionByZero)

    @setup("\\")
    def test_root(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if a >= 0:
                    if b != 0 and (a != 0 or b > 0):
                        self.assertEqual(val, int(a ** (1 / b)))
                        self.assertFlagFalse(rt.Flags.DivisionByZero)
                        self.assertFlagFalse(rt.Flags.ComplexRoot)
                    else:
                        self.assertEqual(val, 0)
                        self.assertFlagTrue(rt.Flags.DivisionByZero)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.ComplexRoot)

    @setup("%")
    def test_mod(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if b != 0:
                    self.assertEqual(val, a % b)
                    self.assertFlagFalse(rt.Flags.DivisionByZero)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.DivisionByZero)

    @setup("|")
    def test_or(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a | b)

    @setup("&")
    def test_and(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a & b)

    @setup("v")
    def test_xor(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, a ^ b)

    @setup(">")
    def test_rshift(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if b >= 0:
                    self.assertEqual(val, a >> b)
                    self.assertFlagFalse(rt.Flags.InvalidValue)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.InvalidValue)

    @setup("<")
    def test_lshift(self):
        for a, b in self.data:
            with self.subTest(a=a, b=b):
                self.state.stack.append(a)
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                if b >= 0:
                    self.assertEqual(val, a << b)
                    self.assertFlagFalse(rt.Flags.InvalidValue)
                else:
                    self.assertEqual(val, 0)
                    self.assertFlagTrue(rt.Flags.InvalidValue)

    @setup("!")
    def test_not(self):
        for a, b in self.data:
            with self.subTest(a=a):
                self.state.stack.append(a)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, ~a)
            with self.subTest(a=b):
                self.state.stack.append(b)
                self.step()
                val = self.state.stack.pop()
                self.assertEqual(val, ~b)


class StackTestCase(StateTestCase):
    def preTest(self):
        self.state.mode = rt.Mode.Stack

    @setup('+15')
    def test_push(self):
        self.step()
        self.assertStack([15])

    @setup('+-15')
    def test_push_neg(self):
        self.step()
        self.assertStack([-15])

    @setup('+')
    def test_push_no(self):
        self.step()
        self.assertStack([0])
        self.assertFlagTrue(rt.Flags.ReadNoDigits)

    @setup('-')
    def test_pop(self):
        self.state.stack.append(15)
        self.step()
        self.assertStack([])

    @setup('*')
    def test_swap(self):
        self.state.stack.extend([15, 5])
        self.step()
        self.assertStack([5, 15])


class HeapTestCase(StateTestCase):
    def preTest(self):
        self.state.mode = rt.Mode.Heap

    @setup('>')
    def test_lshift(self):
        start = randint(-100, 100)
        self.state.pointer = start
        self.step()
        self.assertPointer(start + 1)

    @setup('<')
    def test_rshift(self):
        start = randint(-100, 100)
        self.state.pointer = start
        self.step()
        self.assertPointer(start - 1)

    @setup('#')
    def test_jump(self):
        amount = randint(-100, 100)
        self.state.stack.append(amount)
        self.state.pointer = randint(-100, 100)
        self.step()
        self.assertPointer(amount)

    @setup('*')
    def test_home(self):
        self.state.pointer = randint(-100, 100)
        self.step()
        self.assertPointer(0)

    @setup('+')
    def test_increment(self):
        base = randint(-100, 100)
        if base == -1:
            base = -2
        self.state.heap[0] = base
        self.step()
        self.assertHeap({0: base + 1})

    @setup('-')
    def test_decrement(self):
        base = randint(-100, 100)
        if base == 1:
            base = 2
        self.state.heap[0] = base
        self.step()
        self.assertHeap({0: base - 1})

    @setup('0')
    def test_null(self):
        self.state.heap[0] = randint(-100, 100)
        self.step()
        self.assertHeap({})

    @setup('&')
    def test_clear(self):
        self.state.heap = {key: 1 for key in range(10)}
        self.step()
        self.assertHeap({})


class IoTestCase(StateTestCase):
    def preTest(self):
        self.state.mode = rt.Mode.IO

    @setup('+')
    def test_write(self):
        val = randint(0, 127)
        self.state.stack.append(val)
        self.step()
        self.assertStack([])
        self.assertEqual(self.stdout.getvalue(), chr(val), "Written character was not expected")

    @setup('-', "abc")
    def test_read(self):
        self.step()
        self.step()
        self.step()
        self.assertStack([ord('a'), ord('b'), ord('c')])

    @setup('?', "a")
    def test_test(self):
        self.step()
        self.assertFlagTrue(rt.Flags.ResultFlag)
        self.state.stdin_buffer.read_one()
        self.step()
        self.assertFlagFalse(rt.Flags.ResultFlag)


# noinspection SpellCheckingInspection
class MapTestCase(StateTestCase):
    def preTest(self):
        self.state.mode = rt.Mode.Map

    @setup('+?')
    def test_read(self):
        self.step()
        self.assertStack([ord('?')])

    @setup('- ')
    def test_write(self):
        self.state.stack.append(ord('?'))
        self.step()
        self.assertMap([['-', '?']])

    @setup('*?')
    def test_null(self):
        self.step()
        self.assertMap([['*', ' ']])

    @setup('# \n  ')
    def test_jump(self):
        self.state.stack.extend([1, 1])
        self.step()
        self.assertX(1)
        self.assertY(1)

    @setup('>')
    def test_wincrement(self):
        self.step()
        self.assertW(2)

    @setup('<  ')
    def test_wdecrement(self):
        self.step()
        self.assertW(2)
        self.state.x = 0
        self.step()
        self.assertW(1)
        self.state.x = 0
        self.step()
        self.assertW(1)

    @setup('v')
    def test_hincrement(self):
        self.step()
        self.assertH(2)

    @setup('^\n \n ')
    def test_hdecrement(self):
        self.step()
        self.assertH(2)
        self.state.x = 0
        self.step()
        self.assertH(1)
        self.state.x = 0
        self.step()
        self.assertH(1)

    @setup('W  ')
    def test_readwidth(self):
        self.step()
        self.assertStack([3])

    @setup('H\n \n ')
    def test_readheight(self):
        self.step()
        self.assertStack([3])

    @setup("   \n X \n   ")
    def test_readx(self):
        self.state.x = 1
        self.state.y = 1
        self.step()
        self.assertStack([1])

    @setup("   \n Y \n   ")
    def test_ready(self):
        self.state.x = 1
        self.state.y = 1
        self.step()
        self.assertStack([1])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TraversalTestCase))
    test_suite.addTest(unittest.makeSuite(ComparisonTestCase))
    test_suite.addTest(unittest.makeSuite(OperationTestCase))
    test_suite.addTest(unittest.makeSuite(StackTestCase))
    test_suite.addTest(unittest.makeSuite(FlagsTestCase))
    test_suite.addTest(unittest.makeSuite(HeapTestCase))
    test_suite.addTest(unittest.makeSuite(MapTestCase))
    test_suite.addTest(unittest.makeSuite(IoTestCase))
    return test_suite


if __name__ == '__main__':
    runner = unittest.runner.TextTestRunner()
    runner.run(suite())

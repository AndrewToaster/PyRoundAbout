from __future__ import annotations
from enum import IntFlag, IntEnum, auto
from typing import Final
from random import choice
from utilities import IoBuffer, to_utf8, from_utf8, parse_map, TextReadable, TextWritable


# noinspection PyArgumentList
class Direction(IntFlag):
    Left = auto()
    Right = auto()
    Up = auto()
    Down = auto()


# noinspection PyArgumentList
class Flags:
    Nothing = 0
    ResultFlag = 1
    ComplexRoot = 2
    DivisionByZero = 4
    ResultTruncated = 8
    ReadNoDigits = 16
    InvalidValue = 32
    Utf8Error = 64


# noinspection PyArgumentList
class Mode(IntEnum):
    Traversal = auto()
    ConditionalTraversal = auto()
    Comparison = auto()
    Flags = auto()
    Operation = auto()
    Stack = auto()
    Heap = auto()
    IO = auto()
    Map = auto()


# noinspection PyArgumentList
class Instruction(IntEnum):
    Halt = auto()
    Reset = auto()
    SetConditionalTraversal = auto()
    SetCompare = auto()
    SetFlags = auto()
    SetOperation = auto()
    SetStack = auto()
    SetHeap = auto()
    SetIO = auto()
    SetMap = auto()
    FlowLeft = auto()
    FlowRight = auto()
    FlowUp = auto()
    FlowDown = auto()
    FlowFSReflect = auto()
    FlowBSReflect = auto()
    FlowVReflect = auto()
    FlowHReflect = auto()
    FlowPReflect = auto()
    FlowXReflect = auto()
    FlowSReflect = auto()
    CompareGreaterThan = auto()
    CompareLessThan = auto()
    CompareEqual = auto()
    CompareNotEqual = auto()
    FlagsFalse = auto()
    FlagsTrue = auto()
    FlagsInvert = auto()
    FlagsPush = auto()
    FlagsClear = auto()
    FlagsTest = auto()
    OperationAdd = auto()
    OperationSub = auto()
    OperationMul = auto()
    OperationDiv = auto()
    OperationPow = auto()
    OperationRoot = auto()
    OperationMod = auto()
    OperationOr = auto()
    OperationAnd = auto()
    OperationXor = auto()
    OperationRShift = auto()
    OperationLShift = auto()
    OperationInvert = auto()
    StackPush = auto()
    StackPop = auto()
    StackSwap = auto()
    StackSave = auto()
    StackLoad = auto()
    StackAny = auto()
    StackDupe = auto()
    StackClear = auto()
    HeapRShift = auto()
    HeapLShift = auto()
    HeapJump = auto()
    HeapHome = auto()
    HeapIncrement = auto()
    HeapDecrement = auto()
    HeapNull = auto()
    HeapClear = auto()
    IoWrite = auto()
    IoRead = auto()
    IoAny = auto()
    MapRead = auto()
    MapWrite = auto()
    MapNull = auto()
    MapJump = auto()
    MapWIncrement = auto()
    MapHIncrement = auto()
    MapWDecrement = auto()
    MapHDecrement = auto()
    MapReadWidth = auto()
    MapReadHeight = auto()
    MapReadX = auto()
    MapReadY = auto()
    NOP = auto()


_INSTRUCTION_MAP: Final[dict[Mode, dict[str, Instruction]]] = {
    Mode.Traversal: {
        '@': Instruction.SetConditionalTraversal,
        '?': Instruction.SetCompare,
        '&': Instruction.SetFlags,
        '%': Instruction.SetOperation,
        '=': Instruction.SetStack,
        '[': Instruction.SetHeap,
        '$': Instruction.SetIO,
        '#': Instruction.SetMap,
        '>': Instruction.FlowRight,
        '<': Instruction.FlowLeft,
        'v': Instruction.FlowDown,
        '^': Instruction.FlowUp,
        '/': Instruction.FlowFSReflect,
        '\\': Instruction.FlowBSReflect,
        '|': Instruction.FlowVReflect,
        '-': Instruction.FlowHReflect,
        '+': Instruction.FlowPReflect,
        'x': Instruction.FlowXReflect,
        '*': Instruction.FlowSReflect
    },
    Mode.ConditionalTraversal: None,  # Same as traversal, will be set later
    Mode.Comparison: {
        '>': Instruction.CompareGreaterThan,
        '<': Instruction.CompareLessThan,
        '=': Instruction.CompareEqual,
        '!': Instruction.CompareNotEqual
    },
    Mode.Flags: {
        '|': Instruction.FlagsTrue,
        '&': Instruction.FlagsFalse,
        '^': Instruction.FlagsInvert,
        '>': Instruction.FlagsPush,
        '?': Instruction.FlagsTest,
        '0': Instruction.FlagsClear
    },
    Mode.Operation: {
        '+': Instruction.OperationAdd,
        '-': Instruction.OperationSub,
        '*': Instruction.OperationMul,
        '/': Instruction.OperationDiv,
        '^': Instruction.OperationPow,
        '\\': Instruction.OperationRoot,
        '%': Instruction.OperationMod,
        '|': Instruction.OperationOr,
        '&': Instruction.OperationAnd,
        'v': Instruction.OperationXor,
        '>': Instruction.OperationRShift,
        '<': Instruction.OperationLShift,
        '!': Instruction.OperationInvert,
    },
    Mode.Stack: {
        '+': Instruction.StackPush,
        '-': Instruction.StackPop,
        '*': Instruction.StackSwap,
        '>': Instruction.StackSave,
        '<': Instruction.StackLoad,
        '?': Instruction.StackAny,
        ':': Instruction.StackDupe,
        '&': Instruction.StackClear
    },
    Mode.Heap: {
        '>': Instruction.HeapRShift,
        '<': Instruction.HeapLShift,
        '#': Instruction.HeapJump,
        '*': Instruction.HeapHome,
        '+': Instruction.HeapIncrement,
        '-': Instruction.HeapDecrement,
        '0': Instruction.HeapNull,
        '&': Instruction.HeapClear
    },
    Mode.IO: {
        '+': Instruction.IoWrite,
        '-': Instruction.IoRead,
        '?': Instruction.IoAny
    },
    Mode.Map: {
        '+': Instruction.MapRead,
        '-': Instruction.MapWrite,
        '*': Instruction.MapNull,
        '#': Instruction.MapJump,
        '>': Instruction.MapWIncrement,
        '<': Instruction.MapWDecrement,
        'v': Instruction.MapHIncrement,
        '^': Instruction.MapHDecrement,
        'W': Instruction.MapReadWidth,
        'H': Instruction.MapReadHeight,
        'X': Instruction.MapReadX,
        'Y': Instruction.MapReadY
    }
}
_INSTRUCTION_MAP[Mode.ConditionalTraversal] = _INSTRUCTION_MAP[Mode.Traversal]


class State:
    @property
    def should_flow(self) -> bool:
        return self.mode != Mode.ConditionalTraversal or self.has_flag(Flags.ResultFlag)

    @property
    def stack_any(self) -> bool:
        return len(self.stack) >= 1

    @property
    def stack_two(self) -> bool:
        return len(self.stack) >= 2

    @property
    def heap_has_cell(self) -> bool:
        return self.pointer in self.heap

    @property
    def current_symbol(self) -> str:
        return self.map[self.y][self.x]

    @current_symbol.setter
    def current_symbol(self, value: str):
        self.map[self.y][self.x] = value

    def __init__(self, stack: list[int], heap: dict[int, int], pointer: int, flags: int, mode: Mode,
                 direction: Direction, x: int, y: int, width: int, height: int, map: list[list[str]],
                 stdin: TextReadable, stdout: TextWritable):
        self.stack: list[int] = stack
        self.heap: dict[int, int] = heap
        self.pointer = pointer
        self.flags: int = flags
        self.mode: Mode = mode
        self.direction: Direction = direction
        self.x: int = x
        self.y: int = y
        self.w: int = width
        self.initial_w: Final[int] = width
        self.h: int = height
        self.initial_h: Final[int] = height
        self.map: list[list[str]] = map
        self.initial_map: Final[list[list[str]]] = map
        self.stdin: TextReadable = stdin
        self.stdout: TextWritable = stdout
        self.stdin_buffer: IoBuffer = IoBuffer(stdin)

    def step_cursor(self, x: int = None, y: int = None, relative: bool = True) -> None:
        if not x and not y:
            if Direction.Left in self.direction:
                x = -1
            elif Direction.Right in self.direction:
                x = 1
            else:
                x = 0
            if Direction.Up in self.direction:
                y = -1
            elif Direction.Down in self.direction:
                y = 1
            else:
                y = 0

        if x is None or y is None:
            raise ValueError("Coordinates are invalid")

        self.x = ((self.x + x) if relative else x) % self.w
        self.y = ((self.y + y) if relative else y) % self.h

    def get_instruction(self) -> Instruction:
        sym = self.current_symbol
        if sym == '~':
            return Instruction.Halt
        elif sym == ';':
            return Instruction.Reset
        elif sym in _INSTRUCTION_MAP[self.mode]:
            return _INSTRUCTION_MAP[self.mode][sym]
        else:
            return Instruction.NOP

    def set_flag(self, mask: int, value: bool) -> None:
        self.flags = (self.flags | mask) if value else (self.flags & ~mask)

    def has_flag(self, mask: int) -> bool:
        return self.flags & mask == mask

    def reset(self):
        self.stack.clear()
        self.heap.clear()
        self.x = 0
        self.y = 0
        self.w = self.initial_w
        self.h = self.initial_h
        self.mode = Mode.Traversal
        self.direction = Direction.Right
        self.flags = 0
        self.map = self.initial_map
        self.pointer = 0

    @staticmethod
    def create(content: str, width: int, height: int, stdin: TextReadable, stdout: TextWritable) -> State:
        return State([], {}, 0, 0, Mode.Traversal, Direction.Right, 0, 0, width, height,
                     parse_map(content, width, height), stdin, stdout)


def read_digits(state: State) -> str:
    digits: list[str] = []
    if state.current_symbol == '-':
        digits.append('-')
        state.step_cursor()
    while True:
        sym = state.current_symbol
        if ord('0') <= ord(sym) <= ord('9'):
            digits.append(sym)
            state.step_cursor()
        else:
            break
    if len(digits) == 0:
        state.set_flag(Flags.ReadNoDigits, True)
    return ''.join(digits)


def step_state(state: State) -> bool:
    instr = state.get_instruction()
    step = True

    if instr is Instruction.NOP:
        state.step_cursor()
        return True
    elif instr is Instruction.Halt:
        return False
    elif instr is Instruction.Reset:
        state.mode = Mode.Traversal

    # === Mode Traversal & Flow Testing === #
    elif state.mode is Mode.Traversal or state.mode is Mode.ConditionalTraversal:
        # Can't use break, this will do
        if not state.should_flow:
            pass
        elif instr is Instruction.SetConditionalTraversal:
            state.mode = Mode.ConditionalTraversal
        elif instr is Instruction.SetCompare:
            state.mode = Mode.Comparison
        elif instr is Instruction.SetFlags:
            state.mode = Mode.Flags
        elif instr is Instruction.SetOperation:
            state.mode = Mode.Operation
        elif instr is Instruction.SetStack:
            state.mode = Mode.Stack
        elif instr is Instruction.SetHeap:
            state.mode = Mode.Heap
        elif instr is Instruction.SetIO:
            state.mode = Mode.IO
        elif instr is Instruction.SetMap:
            state.mode = Mode.Map
        elif instr is Instruction.FlowUp:
            state.direction = Direction.Up
        elif instr is Instruction.FlowDown:
            state.direction = Direction.Down
        elif instr is Instruction.FlowLeft:
            state.direction = Direction.Left
        elif instr is Instruction.FlowRight:
            state.direction = Direction.Right
        elif instr is Instruction.FlowFSReflect:
            if state.direction is Direction.Right or state.direction is Direction.Up:
                state.direction = Direction.Right | Direction.Up
            elif state.direction is Direction.Left or state.direction is Direction.Down:
                state.direction = Direction.Left | Direction.Down
            elif state.direction is Direction.Left | Direction.Up:
                state.direction = Direction.Right | Direction.Down
            elif state.direction is Direction.Right | Direction.Down:
                state.direction = Direction.Left | Direction.Up
        elif instr is Instruction.FlowBSReflect:
            if state.direction is Direction.Left or state.direction is Direction.Up:
                state.direction = Direction.Left | Direction.Up
            elif state.direction is Direction.Right or state.direction is Direction.Down:
                state.direction = Direction.Right | Direction.Down
            elif state.direction is Direction.Right | Direction.Up:
                state.direction = Direction.Left | Direction.Down
            elif state.direction is Direction.Left | Direction.Down:
                state.direction = Direction.Right | Direction.Up
        elif instr is Instruction.FlowVReflect:
            if state.direction is Direction.Right:
                state.direction = Direction.Left
            elif state.direction is Direction.Left:
                state.direction = Direction.Right
            elif Direction.Up in state.direction:
                state.direction = Direction.Up
            elif Direction.Down in state.direction:
                state.direction = Direction.Down
        elif instr is Instruction.FlowHReflect:
            if state.direction is Direction.Up:
                state.direction = Direction.Down
            elif state.direction is Direction.Down:
                state.direction = Direction.Up
            elif Direction.Left in state.direction:
                state.direction = Direction.Left
            elif Direction.Right in state.direction:
                state.direction = Direction.Right
        elif instr is Instruction.FlowPReflect:
            state.direction = choice([
                Direction.Left,
                Direction.Right,
                Direction.Up,
                Direction.Down
            ])
        elif instr is Instruction.FlowXReflect:
            state.direction = choice([
                Direction.Left | Direction.Up,
                Direction.Left | Direction.Down,
                Direction.Right | Direction.Up,
                Direction.Right | Direction.Down
            ])
        elif instr is Instruction.FlowSReflect:
            state.direction = choice([
                Direction.Left,
                Direction.Right,
                Direction.Up,
                Direction.Down,
                Direction.Left | Direction.Up,
                Direction.Left | Direction.Down,
                Direction.Right | Direction.Up,
                Direction.Right | Direction.Down
            ])

    # === Mode Comparison === #
    elif state.mode is Mode.Comparison:
        if instr is Instruction.CompareGreaterThan:
            if state.stack_two:
                state.set_flag(Flags.ResultFlag, state.stack[-2] > state.stack[-1])
        elif instr is Instruction.CompareLessThan:
            if state.stack_two:
                state.set_flag(Flags.ResultFlag, state.stack[-2] < state.stack[-1])
        elif instr is Instruction.CompareEqual:
            if state.stack_two:
                state.set_flag(Flags.ResultFlag, state.stack[-2] == state.stack[-1])
        elif instr is Instruction.CompareNotEqual:
            if state.stack_two:
                state.set_flag(Flags.ResultFlag, state.stack[-2] != state.stack[-1])

    # === Mode Flags === #
    elif state.mode is Mode.Flags:
        if instr is Instruction.FlagsTrue:
            if state.stack_any:
                value = state.stack.pop()
                if value >= 0:
                    state.flags |= value
                else:
                    state.flags |= Flags.InvalidValue
        elif instr is Instruction.FlagsFalse:
            if state.stack_any:
                value = state.stack.pop()
                if value >= 0:
                    state.flags &= ~value
                else:
                    state.flags |= Flags.InvalidValue
        elif instr is Instruction.FlagsInvert:
            if state.stack_any:
                value = state.stack.pop()
                if value >= 0:
                    state.flags = state.flags ^ value
                else:
                    state.flags |= Flags.InvalidValue
        elif instr is Instruction.FlagsTest:
            if state.stack_any:
                value = state.stack.pop()
                if value >= 0:
                    state.set_flag(Flags.ResultFlag, state.flags & value == value)
                else:
                    state.flags |= Flags.InvalidValue
        elif instr is Instruction.FlagsPush:
            state.stack.append(state.flags)
        elif instr is Instruction.FlagsClear:
            state.flags = 0

    # === Mode Operation === #
    elif state.mode is Mode.Operation:
        if instr is Instruction.OperationAdd:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left + op_right)
        elif instr is Instruction.OperationSub:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left - op_right)
        elif instr is Instruction.OperationMul:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left * op_right)
        elif instr is Instruction.OperationDiv:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                if op_right != 0:
                    result, rem = divmod(op_left, op_right)
                    state.stack.append(result)
                    state.set_flag(Flags.ResultTruncated, rem != 0)
                    state.set_flag(Flags.DivisionByZero, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.ResultTruncated, False)
                    state.set_flag(Flags.DivisionByZero, True)
        elif instr is Instruction.OperationPow:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()

                if not (op_left == 0 and op_right < 1):
                    value = op_left ** op_right
                    actual = int(value)
                    state.stack.append(actual)
                    state.set_flag(Flags.ResultTruncated, value != actual)
                    state.set_flag(Flags.DivisionByZero, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.ResultTruncated, False)
                    state.set_flag(Flags.DivisionByZero, True)
        elif instr is Instruction.OperationRoot:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                if op_left >= 0:
                    if op_right != 0 and (op_left != 0 or op_right > 0):
                        state.stack.append(int(op_left ** (1 / op_right)))
                        state.set_flag(Flags.DivisionByZero, False)
                        state.set_flag(Flags.ComplexRoot, False)
                    else:
                        state.stack.append(0)
                        state.set_flag(Flags.DivisionByZero, True)
                        state.set_flag(Flags.ComplexRoot, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.DivisionByZero, False)
                    state.set_flag(Flags.ComplexRoot, True)
        elif instr is Instruction.OperationMod:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                if op_right != 0:
                    state.stack.append(op_left % op_right)
                    state.set_flag(Flags.DivisionByZero, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.DivisionByZero, True)
        elif instr is Instruction.OperationOr:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left | op_right)
        elif instr is Instruction.OperationAnd:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left & op_right)
        elif instr is Instruction.OperationXor:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                state.stack.append(op_left ^ op_right)
        elif instr is Instruction.OperationRShift:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                if op_right >= 0:
                    state.stack.append(op_left >> op_right)
                    state.set_flag(Flags.InvalidValue, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.InvalidValue, True)
        elif instr is Instruction.OperationLShift:
            if state.stack_two:
                op_right: int = state.stack.pop()
                op_left: int = state.stack.pop()
                if op_right >= 0:
                    state.stack.append(op_left << op_right)
                    state.set_flag(Flags.InvalidValue, False)
                else:
                    state.stack.append(0)
                    state.set_flag(Flags.InvalidValue, True)
        elif instr is Instruction.OperationInvert:
            if state.stack_any:
                value: int = state.stack.pop()
                state.stack.append(~value)

    # === Mode Stack === #
    elif state.mode is Mode.Stack:
        if instr is Instruction.StackPush:
            state.step_cursor()
            step = False
            digits = read_digits(state)
            if len(digits) != 0:
                state.stack.append(int(digits))
            else:
                state.stack.append(0)
        elif instr is Instruction.StackPop:
            if state.stack_any:
                state.stack.pop()
        elif instr is Instruction.StackSwap:
            if state.stack_two:
                (state.stack[-1], state.stack[-2]) = (state.stack[-2], state.stack[-1])
        elif instr is Instruction.StackSave:
            if state.stack_any:
                value: int = state.stack.pop()
                if value != 0:
                    state.heap[state.pointer] = value
                else:
                    del state.heap[state.pointer]
        elif instr is Instruction.StackLoad:
            if state.heap_has_cell:
                state.stack.append(state.heap[state.pointer])
            else:
                state.stack.append(0)
        elif instr is Instruction.StackAny:
            state.set_flag(Flags.ResultFlag, len(state.stack) > 0)
        elif instr is Instruction.StackDupe:
            if state.stack_any:
                state.stack.append(state.stack[-1])
        elif instr is Instruction.StackClear:
            state.stack.clear()

    # === Mode Heap === #
    elif state.mode is Mode.Heap:
        if instr is Instruction.HeapRShift:
            state.pointer += 1
        elif instr is Instruction.HeapLShift:
            state.pointer -= 1
        elif instr is Instruction.HeapJump:
            if state.stack_any:
                state.pointer = state.stack.pop()
        elif instr is Instruction.HeapHome:
            state.pointer = 0
        elif instr is Instruction.HeapIncrement:
            if state.heap_has_cell:
                state.heap[state.pointer] += 1
            else:
                state.heap[state.pointer] = 1
        elif instr is Instruction.HeapDecrement:
            if state.heap_has_cell:
                state.heap[state.pointer] -= 1
            else:
                state.heap[state.pointer] = -1
        elif instr is Instruction.HeapNull:
            if state.heap_has_cell:
                del state.heap[state.pointer]
        elif instr is Instruction.HeapClear:
            state.heap.clear()

    # === Mode IO === #
    elif state.mode is Mode.IO:
        if instr is Instruction.IoWrite:
            if state.stack_any:
                value: int = state.stack.pop()
                try:
                    state.stdout.write(to_utf8(value))
                    state.stdout.flush()
                    state.set_flag(Flags.Utf8Error, False)
                except UnicodeError:
                    state.set_flag(Flags.Utf8Error, True)

        elif instr is Instruction.IoRead:
            if state.stdin_buffer.any():
                try:
                    state.stack.append(from_utf8(state.stdin_buffer.read_one()))
                    state.set_flag(Flags.Utf8Error, False)
                except UnicodeError:
                    state.set_flag(Flags.Utf8Error, True)
            else:
                state.stack.append(-1)
        elif instr is Instruction.IoAny:
            state.set_flag(Flags.ResultFlag, state.stdin_buffer.any())

    # === Mode Map === #
    elif state.mode is Mode.Map:
        if instr is Instruction.MapRead:
            state.step_cursor()
            try:
                state.stack.append(from_utf8(state.current_symbol))
                state.set_flag(Flags.Utf8Error, False)
            except UnicodeError:
                state.set_flag(Flags.Utf8Error, True)
        elif instr is Instruction.MapWrite:
            if state.stack_any:
                state.step_cursor()
                try:
                    state.current_symbol = to_utf8(state.stack.pop())
                    state.set_flag(Flags.Utf8Error, False)
                except UnicodeError:
                    state.set_flag(Flags.Utf8Error, True)
        if instr is Instruction.MapNull:
            state.step_cursor()
            state.current_symbol = ' '
        elif instr is Instruction.MapJump:
            if state.stack_two:
                (y, x) = (state.stack.pop(), state.stack.pop())
                state.step_cursor(x, y, False)
                step = False
        elif instr is Instruction.MapWIncrement:
            state.w += 1
            for line in state.map:
                line.append(' ')
        elif instr is Instruction.MapWDecrement:
            if state.w > 1:
                state.w -= 1
                for line in state.map:
                    del line[-1]
        elif instr is Instruction.MapHIncrement:
            state.h += 1
            state.map.append([' '] * state.w)
        elif instr is Instruction.MapHDecrement:
            if state.h > 1:
                state.h -= 1
                del state.map[-1]
        elif instr is Instruction.MapReadWidth:
            state.stack.append(state.w)
        elif instr is Instruction.MapReadHeight:
            state.stack.append(state.h)
        elif instr is Instruction.MapReadX:
            state.stack.append(state.x)
        elif instr is Instruction.MapReadY:
            state.stack.append(state.y)
    else:
        pass

    if step:
        state.step_cursor()
    return True

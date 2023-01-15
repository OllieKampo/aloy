import dis
from typing import Callable, Iterable

def list_func_calls(fn):
    funcs = []
    bytecode = dis.Bytecode(fn)
    instrs = list(reversed([instr for instr in bytecode]))
    for (ix, instr) in enumerate(instrs):
        if str(instr.opname).startswith("CALL_FUNCTION"):
            load_func_instr = instrs[ix + instr.arg + 1]
            funcs.append(load_func_instr.argval)

    return ["%d. %s" % (ix, funcname) for (ix, funcname) in enumerate(reversed(funcs), 1)]

def loads_functions(func: Callable, func_names: Iterable[str]) -> Iterable[str]:
    """Return a list of the given function names that are loaded by the given function object."""
    loads: list[str] = []
    instructions = list(dis.get_instructions(func))
    for instruction in instructions:
        if (str(instruction.opname) in ["LOAD_GLOBAL", "LOAD_METHOD"]
            and (func_name := str(instruction.argval)) in func_names):
            loads.append(func_name)
    return loads
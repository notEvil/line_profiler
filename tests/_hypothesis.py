import hypothesis.strategies as h_strategies
import dataclasses
import enum
import re


_NO_WORK = "None"


@h_strategies.composite
def code_strings(
    draw,
    max_functions=4,
    max_statements=4,
    max_elements=4,
    max_work=4,
    max_entries=4,
    max_depth=4,
):
    async_ = draw(h_strategies.booleans())

    # function properties
    _v_ = h_strategies.builds(
        _FunctionProperties,
        profile=h_strategies.booleans(),
        async_=h_strategies.booleans() if async_ else h_strategies.just(False),
        generator=h_strategies.booleans(),
    )
    _v_ = draw(h_strategies.lists(_v_, min_size=1, max_size=max_functions))
    function_properties_list = _v_

    for index, function_properties in enumerate(function_properties_list):
        function_properties.name = f"function_{index + 1}"
    #

    entry_names = [
        function_properties.name
        for function_properties in function_properties_list
        if function_properties.profile
        and not (async_ and not function_properties.async_)
        and not function_properties.generator
    ]
    if len(entry_names) == 0:
        function_properties = function_properties_list[0]
        function_properties.profile = True
        function_properties.async_ = async_
        function_properties.generator = False

        entry_names.append(function_properties.name)

    _v_ = [
        _functions(
            function_properties,
            function_properties_list,
            max_statements,
            max_elements,
            max_work,
        )
        for function_properties in function_properties_list
    ]
    _v_ = h_strategies.tuples(
        h_strategies.tuples(*_v_),
        h_strategies.lists(
            _entries(entry_names, max_depth), min_size=1, max_size=max_entries
        ),
    )
    function_strings, entry_strings = draw(_v_)

    functions_string = "\n\n\n".join(function_strings)

    if async_:
        _v_ = (f"task_group.create_task({string})" for string in entry_strings)
        string = _indent("\n".join(_v_), 8)

        entries_string = f"""
async def async_main():
    async with asyncio.TaskGroup() as task_group:
{string}


asyncio.run(async_main())
"""[1:-1]

    else:
        entries_string = "\n".join(entry_strings)

    async_string = "import asyncio\n\n\n" if async_ else ""
    return f"""
{async_string}{functions_string}


{entries_string}
"""[1:-1]


@dataclasses.dataclass
class _FunctionProperties:
    profile: bool | None = None
    async_: bool | None = None
    name: str | None = None
    generator: bool | None = None


@h_strategies.composite
def _functions(
    draw,
    function_properties,
    function_properties_list,
    max_statements,
    max_elements,
    max_work,
):
    profile_string = "@profile\n" if function_properties.profile else ""
    async_string = "async " if function_properties.async_ else ""

    _v_ = lambda _function_properties: not (
        not function_properties.async_ and _function_properties.async_
    )
    _v_ = list(filter(_v_, function_properties_list))
    _v_ = _statements(function_properties, _v_, max_elements, max_work)
    _v_ = "\n".join(draw(h_strategies.lists(_v_, min_size=1, max_size=max_statements)))
    statements_string = _indent(_v_, 8)

    _v_ = function_properties.generator
    generator_string = _indent("\n\nif line(False):\n    yield", 4) if _v_ else ""

    return f"""
{profile_string}{async_string}def {function_properties.name}(depth):
    if line(0 < depth):
{statements_string}{generator_string}
"""[1:-1]


def _indent(string, number):
    indent_string = " " * number
    string = re.sub(r"\n(?!\n)", f"\n{indent_string}", string)
    return string if string.startswith("\n") else f"{indent_string}{string}"


class _StatementType(enum.Enum):
    default = enum.auto()
    yield_ = enum.auto()
    yield_from = enum.auto()


@h_strategies.composite
def _statements(
    draw, function_properties, function_properties_list, max_elements, max_work
):
    statement_types = [_StatementType.default]

    if function_properties.generator:
        statement_types.append(_StatementType.yield_)

    if function_properties.generator:
        generator_function_properties_list = [
            function_properties
            for function_properties in function_properties_list
            if function_properties.generator
        ]

        if len(generator_function_properties_list) != 0:
            statement_types.append(_StatementType.yield_from)

    else:
        generator_function_properties_list = None

    match draw(h_strategies.sampled_from(statement_types)):
        case _StatementType.default:
            strategies = [_works(function_properties.async_, max_work)]

            if len(function_properties_list) != 0:
                strategies.append(_calls(function_properties_list))

            _v_ = h_strategies.one_of(*strategies)
            _v_ = h_strategies.tuples(h_strategies.booleans(), _v_)
            tuples = draw(h_strategies.lists(_v_, min_size=1, max_size=max_elements))

            _, string = tuples[0]
            tuples[0] = (False, string)

            if all(string == _NO_WORK for _, string in tuples):
                _v_ = (
                    f"\n {_NO_WORK}" if newline else f" {_NO_WORK}"
                    for newline, _ in tuples
                )
                string = ",".join(_v_).removeprefix(" ")

                return f"line({string})"

            index = 0
            strings = []

            for newline, string in tuples:
                if newline:
                    index = 0

                strings.append(",\n " if newline else ", ")
                strings.append(f"line({string})" if index == 0 else string)

                index += 1

            string = "".join(strings[1:])

            _v_ = f"line({string})" if any(newline for newline, _ in tuples) else string
            return _v_

        case _StatementType.yield_:
            return "yield line()"

        case _StatementType.yield_from:
            _v_ = draw(_generators(generator_function_properties_list))
            string, generator_function_properties = _v_

            if function_properties.async_:
                iter_name = "aiter" if generator_function_properties.async_ else "iter"
                async_string = "await a" if generator_function_properties.async_ else ""
                return f"""
iterator = line({iter_name}({string}))
while line({async_string}next(iterator, NONE) is not NONE):
    yield line()
"""[1:-1]  # can't use for because of line counts

            return f"yield from line({string})"


@h_strategies.composite
def _generators(draw, function_properties_list):
    _v_ = h_strategies.sampled_from(("depth - 1", "depth // 2"))
    _v_ = h_strategies.tuples(h_strategies.sampled_from(function_properties_list), _v_)
    function_properties, depth_string = draw(_v_)

    return (f"{function_properties.name}({depth_string})", function_properties)


@h_strategies.composite
def _works(draw, async_, max_work):
    _v_ = h_strategies.booleans() if async_ else h_strategies.just(False)
    async_, time = draw(
        h_strategies.tuples(_v_, h_strategies.integers(min_value=0, max_value=max_work))
    )

    if time == 0 and not async_:
        return _NO_WORK

    return f"await async_work({time})" if async_ else f"work({time})"


@h_strategies.composite
def _calls(draw, function_properties_list):
    _v_ = h_strategies.sampled_from(("depth - 1", "depth // 2"))
    _v_ = h_strategies.tuples(h_strategies.sampled_from(function_properties_list), _v_)
    function_properties, depth_string = draw(_v_)

    call_string = f"{function_properties.name}({depth_string})"

    if function_properties.generator:
        return (
            f"[line12(_) async for _ in {call_string}]"
            if function_properties.async_
            else f"list({call_string})"
        )

    return f"await {call_string}" if function_properties.async_ else call_string


@h_strategies.composite
def _entries(draw, names, max_depth):
    _v_ = h_strategies.integers(min_value=0, max_value=max_depth)
    name, depth = draw(h_strategies.tuples(h_strategies.sampled_from(names), _v_))

    return f"{name}({depth})"

import typer
import pathlib
import pickle
import re


def main(path: pathlib.Path):
    with open(path, "rb") as file:
        line_stats = pickle.load(file)

    node_ids = {}
    edges = set()

    node_times = {}
    edge_times = {}
    max_time = 0

    for caller_key, sub_dict in line_stats.call_stats.items():
        for sub_dict in sub_dict.values():
            for callee_key, sub_dict in sub_dict.items():
                _v_ = (call_stats.total_stats.sum_1 for call_stats in sub_dict.values())
                total_time = sum(_v_) * line_stats.unit

                node_times[callee_key] = node_times.get(callee_key, 0) + total_time

                _v_ = sub_dict.values()
                _v_ = sum(call_stats.cumulative_stats.sum_1 for call_stats in _v_)
                cumulative_time = _v_ * line_stats.unit

                edge = (caller_key, callee_key)
                edge_times[edge] = edge_times.get(edge, 0) + cumulative_time

                if caller_key is None:
                    max_time += cumulative_time

    node_times[None] = max_time
    edge_times[None] = max_time

    print("digraph {")
    print("node [ style=filled ]")

    for caller_key, sub_dict in line_stats.call_stats.items():
        caller_node = _get_node(caller_key, node_ids, node_times)

        for caller_line, sub_dict in sub_dict.items():
            for callee_key, sub_dict in sub_dict.items():
                callee_node = _get_node(callee_key, node_ids, node_times)
                edge = (caller_node, callee_node)
                if edge in edges:
                    continue

                seconds = edge_times[(caller_key, callee_key)]

                number = seconds / edge_times[None]
                number = round(number * 255 * 0 + (1 - number) * 255 * 0.9)
                print(
                    f"{caller_node} -> {callee_node} ["
                    f' label="{_get_time_string(seconds)}"'
                    f' color="#{number:02x}{number:02x}{number:02x}" ]'
                )
                edges.add(edge)

    print("}")


def _get_node(key, node_ids, node_times):
    node_id = node_ids.get(key)
    if node_id is not None:
        return node_id

    if key is None:
        string = "entry"

    else:
        path_string, line_number, name = key
        string = f'{_get_module_string(path_string)}:{line_number} "{name}"'

    label = re.sub(r'"', r"\"", f"{string} ({_get_time_string(node_times[key])})")

    node_id = f"n{len(node_ids)}"

    number = node_times[key] / node_times[None]
    number = round(number * 255 * 0.1 + (1 - number) * 255 * 1)
    print(f'{node_id} [ label="{label}" fillcolor="#ff{number:02x}{number:02x}" ]')

    node_ids[key] = node_id
    return node_id


def _get_module_string(path_string):
    path = pathlib.Path(path_string)
    if path.name == "__init__.py":
        path = path.parent

    names = [path.name]
    while (path.parent / "__init__.py").exists():
        path = path.parent
        names.append(path.name)

    return ".".join(reversed(names))


def _get_time_string(seconds):
    return f"{int(seconds * 1e3)} ms" if seconds < 1 else f"{seconds:0.1f} s"


typer.run(main)

import typer
import colorsys
import pathlib
import pickle
import re


def interpolate_linear(number, first_number, second_number):
    return (1 - number) * first_number + number * second_number


def get_rgb(l, r, g, b):
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s)


def main(path: pathlib.Path):
    with open(path, "rb") as file:
        line_stats = pickle.load(file)

    # get times
    node_times = {
        key: sum(timing.total_time for timing in timings) * line_stats.unit
        for key, timings in line_stats.timings.items()
    }

    edge_times = {}
    for caller_key, sub_dict in line_stats.call_times.items():
        for cumulative_times in sub_dict.values():
            for callee_key, cumulative_time in cumulative_times.items():
                edge_key = (caller_key, callee_key)

                _v_ = edge_times.get(edge_key, 0) + cumulative_time * line_stats.unit
                edge_times[edge_key] = _v_

    _v_ = line_stats.call_times.get(None, {}).get(None, {}).values()
    max_time = sum(cumulative_time for cumulative_time in _v_) * line_stats.unit

    node_times[None] = max_time
    edge_times[None] = max_time
    #

    node_ids = {}
    edge_keys = set()
    edge_tuples = []

    for caller_key, sub_dict in line_stats.call_times.items():
        for callee_keys in sub_dict.values():
            for callee_key in callee_keys:
                edge_key = (caller_key, callee_key)

                if edge_key in edge_keys:
                    continue
                edge_keys.add(edge_key)

                seconds = edge_times[edge_key]
                edge_tuples.append((seconds, callee_key, caller_key))

    edge_tuples.sort(reverse=True)

    print("digraph {")
    print("node [ style=filled ]")

    for seconds, callee_key, caller_key in edge_tuples:
        caller_node = _get_node(caller_key, node_ids, node_times)
        callee_node = _get_node(callee_key, node_ids, node_times)
        print(
            f'{caller_node} -> {callee_node} [ label="{_get_time_string(seconds)}"'
            f' color="{_get_color(interpolate_linear(seconds / edge_times[None], 0.8, 0), 0.5, 0.5, 0.5)}"'
            f" penwidth={interpolate_linear(seconds / edge_times[None], 1, 5)} ]"
        )

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

    _v_ = interpolate_linear(node_times[key] / node_times[None], 1, 0.40)
    print(f'{node_id} [ label="{label}" fillcolor="{_get_color( _v_, 1, 0, 0 )}" ]')

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


def _get_color(number, r, g, b):
    r, g, b = get_rgb(number, r, g, b)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


if __name__ == "__main__":
    typer.run(main)

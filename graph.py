import line_profiler._line_profiler as l__line_profiler
import typer
import colorsys
import pathlib
import pickle
import re


def all_equal(objects):
    objects = iter(objects)
    first_object = next(objects, None)
    return all(object == first_object for object in objects)


def interpolate_linear(number, first_number, second_number):
    return (1 - number) * first_number + number * second_number


def get_rgb(l, r, g, b):
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s)


def main(path: pathlib.Path, remove_contextlib: bool = True):
    with open(path, "rb") as file:
        line_stats = pickle.load(file)

    # get times
    node_stats = {
        key: _NodeStats(
            block_profile.hit_count,
            block_profile.primitive_hit_count,
            sum(
                line_profile.total_time
                for line_profile in line_stats.line_profiles.get(key, [])
            ),
            block_profile.cumulative_time,
        )
        for key, block_profile in line_stats.block_profiles.items()
    }

    call_profiles: dict = {
        (caller_key, callee_key): call_profile
        for caller_key, sub_dict in line_stats.call_profiles.items()
        for callee_key, call_profile in sub_dict.items()
    }

    _v_ = line_stats.call_profiles.get(None, {}).values()
    _v_ = sum(call_profile.cumulative_time for call_profile in _v_)
    max_call_profile = l__line_profiler.CallProfile(None, None, _v_)

    call_profiles[None] = max_call_profile
    node_stats[None] = max_call_profile
    #

    node_ids = {}

    print("digraph {")
    print("node [ style=filled ]")

    _v_ = lambda tuple: tuple[1].cumulative_time
    for tuple, call_profile in sorted(call_profiles.items(), key=_v_, reverse=True):
        if tuple is None:
            continue

        caller_key, callee_key = tuple

        _v_ = _get_node(caller_key, node_ids, node_stats, line_stats, remove_contextlib)
        caller_node = _v_

        _v_ = _get_node(callee_key, node_ids, node_stats, line_stats, remove_contextlib)
        callee_node = _v_
        if callee_node is None:
            continue

        print(
            f"{caller_node} ->"
            f' {callee_node} [label="{_get_hit_string([call_profile.primitive_hit_count, call_profile.hit_count])} |'
            f' {_get_time_string([call_profile.cumulative_time * line_stats.unit])}"'
            f' color="{_get_color(interpolate_linear(call_profile.cumulative_time / call_profiles[None].cumulative_time, 0.8, 0), 0.5, 0.5, 0.5)}"'
            f" penwidth={interpolate_linear(call_profile.cumulative_time / call_profiles[None].cumulative_time, 1, 5)}]"
        )

    print("}")


class _NodeStats:
    def __init__(self, hit_count, primitive_hit_count, total_time, cumulative_time):
        super().__init__()

        self.hit_count = hit_count
        self.primitive_hit_count = primitive_hit_count
        self.total_time = total_time
        self.cumulative_time = cumulative_time


def _get_node(key, node_ids, node_stats, line_stats, remove_contextlib):
    node_id = node_ids.get(key)
    if node_id is not None:
        return node_id

    object = node_stats[key]
    if key is None:
        name = "entry"
        total_time = object.cumulative_time

        _v_ = f"entry | {_get_time_string([object.cumulative_time * line_stats.unit])}"
        label = _v_

    else:
        path_string, line_number, name = key
        module_string = _get_module_string(path_string)
        if remove_contextlib and module_string == "contextlib.py":
            return

        total_time = object.total_time
        label = (
            f"{module_string}:{line_number} | {name} |"
            f" {_get_hit_string([object.primitive_hit_count, object.hit_count])} |"
            f" {_get_time_string([total_time * line_stats.unit, object.cumulative_time * line_stats.unit])}"
        )

    node_id = f"n{len(node_ids)}"
    label = re.sub(r'"', r"\"", label)

    _v_ = interpolate_linear(total_time / node_stats[None].cumulative_time, 1, 0.40)
    print(f'{node_id} [ label="{label}" fillcolor="{_get_color(_v_, 1, 0, 0)}" ]')

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


def _get_hit_string(hits_list):
    if all_equal(hits_list):
        hits_list = hits_list[:1]

    string = "/".join(map(str, hits_list))
    return f"{string}x"


def _get_time_string(seconds_list):
    if all_equal(seconds_list):
        seconds_list = seconds_list[:1]

    ms = all(seconds < 1 for seconds in seconds_list)
    string = "/".join(
        [f"{int(seconds * 1e3)}" for seconds in seconds_list]
        if ms
        else [f"{seconds:0.1f}" for seconds in seconds_list]
    )
    return f"{string}ms" if ms else f"{string}s"


def _get_color(number, r, g, b):
    r, g, b = get_rgb(number, r, g, b)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


if __name__ == "__main__":
    typer.run(main)

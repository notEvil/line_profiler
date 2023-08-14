import typer
import pathlib
import pickle
import re


def interpolate_linear(number, first_number, second_number):
    return (1 - number) * first_number + number * second_number


class Rec709Colors:
    """
    - provides a linear color space based on the model for perceived lightness of Rec. 709
      - y(rgb) = 0.2126 * r + 0.7152 * g + 0.0722 * b
      - y(.get(a)) - y(.get(b)) == (a - b) * C
        - C = y(max_rgb) - y(min_rgb)
    """

    def __init__(self, r, g, b, min_r=0, max_r=1, min_g=0, max_g=1, min_b=0, max_b=1):
        super().__init__()

        self.r = r
        self.g = g
        self.b = b
        self.min_r = min_r
        self.max_r = max_r
        self.min_g = min_g
        self.max_g = max_g
        self.min_b = min_b
        self.max_b = max_b

        self._ys = self._get_ys()
        self._min_y = min(y for y, _ in self._ys)
        self._max_y = max(y for y, _ in self._ys)

    def _get_ys(self):
        ys = []
        for d in [
            self.min_r - self.r,
            self.min_g - self.g,
            self.min_b - self.b,
            self.max_r - self.r,
            self.max_g - self.g,
            self.max_b - self.b,
        ]:
            _v_ = max(self.min_r, min(self.r + d, self.max_r)) * 0.2126
            _v_ = _v_ + max(self.min_g, min(self.g + d, self.max_g)) * 0.7152
            ys.append((_v_ + max(self.min_b, min(self.b + d, self.max_b)) * 0.0722, d))

        ys.sort()
        return ys

    def get(self, number):
        y = interpolate_linear(number, self._min_y, self._max_y)

        for (y1, d1), (y2, d2) in zip(self._ys, self._ys[1:]):
            if y1 <= y and y < y2:
                break

        else:
            y1, d1 = self.ys[-1]
            y2, d2 = y1, d1

        d = d1 if y1 == y2 else interpolate_linear((y - y1) / (y2 - y1), d1, d2)
        return (
            max(self.min_r, min(self.r + d, self.max_r)),
            max(self.min_g, min(self.g + d, self.max_g)),
            max(self.min_b, min(self.b + d, self.max_b)),
        )


def main(path: pathlib.Path):
    with open(path, "rb") as file:
        line_stats = pickle.load(file)

    # get times
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

                edge_key = (caller_key, callee_key)
                edge_times[edge_key] = edge_times.get(edge_key, 0) + cumulative_time

                if caller_key is None:
                    max_time += cumulative_time

    node_times[None] = max_time
    edge_times[None] = max_time
    #

    node_ids = {}
    edge_keys = set()
    edge_tuples = []

    for caller_key, sub_dict in line_stats.call_stats.items():
        for caller_line, sub_dict in sub_dict.items():
            for callee_key, sub_dict in sub_dict.items():
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

    _v_ = interpolate_linear(node_times[key] / node_times[None], 1, 0.20)
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
    r, g, b = Rec709Colors(r, g, b).get(number)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


if __name__ == "__main__":
    typer.run(main)

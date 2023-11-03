from . import _hypothesis
from . import _profiler
import hypothesis
import line_profiler
import time


# TODO
hypothesis.settings.register_profile("test", print_blob=True)
hypothesis.settings.load_profile("test")


class _NONE:
    pass


reproduce_failure = hypothesis.reproduce_failure


@hypothesis.settings(deadline=None)
@hypothesis.given(_hypothesis.code_strings())
def test_hypothesis(code_string):
    debug = False

    if debug:
        _v_ = enumerate(code_string.split("\n"))
        print("\n".join(f"{index + 1:3}| {string}" for index, string in _v_))

    line_profiler_ = _profiler.LineProfiler(debug=debug)

    _v_ = dict(
        NONE=_NONE,
        profile=line_profiler_.profile,
        line=line_profiler_.line,
        line12=line_profiler_.line12,
        work=line_profiler_.work,
        async_work=line_profiler_.async_work,
    )
    exec(compile(code_string, "<string>", "exec"), _v_)

    expected = line_profiler_

    for unit_seconds in (1e-3, 1e-2, 1e-1):
        c_code_string = code_string

        for string, with_string in [
            ("line12", ""),
            ("line", ""),
            ("async_work(", f"asyncio.sleep({unit_seconds} * "),
            ("work(", f"_sleep({unit_seconds} * "),
        ]:
            c_code_string = c_code_string.replace(string, with_string)

        if debug:
            _v_ = enumerate(c_code_string.split("\n"))
            print("\n".join(f"{index + 1:3}| {string}" for index, string in _v_))

        code_object = compile(c_code_string, "<string>", "exec")

        line_profiler_ = line_profiler.LineProfiler()
        exec(code_object, dict(NONE=_NONE, profile=line_profiler_, _sleep=_sleep))
        line_stats = line_profiler_.get_stats()

        try:
            # line hit counts
            _v_ = {
                line_profile.line_number: line_profile.hit_count
                for line_profiles in line_stats.line_profiles.values()
                for line_profile in line_profiles
            }
            assert _without_zeros(_v_) == _without_zeros(expected._line_hit_counts)

            # line primitive hit counts
            _v_ = {
                line_profile.line_number: line_profile.primitive_hit_count
                for line_profiles in line_stats.line_profiles.values()
                for line_profile in line_profiles
            }
            _v_ = _without_zeros(_v_)
            assert _v_ == _without_zeros(expected._line_primitive_hit_counts)

            # line total times
            _v_ = expected._line_total_times.items()
            _v_ = {line_number: time * unit_seconds for line_number, time in _v_}
            expected_total_times = _v_

            _v_ = {
                line_profile.line_number: _Time(
                    line_profile.total_time * line_stats.unit, unit_seconds
                )
                for line_profiles in line_stats.line_profiles.values()
                for line_profile in line_profiles
            }
            assert _without_zeros(_v_) == expected_total_times

            # line cumulative times
            _v_ = expected._line_cumulative_times.items()
            _v_ = {line_number: time * unit_seconds for line_number, time in _v_}
            expected_cumulative_times = _v_

            _v_ = {
                line_profile.line_number: _Time(
                    line_profile.cumulative_time * line_stats.unit, unit_seconds
                )
                for line_profiles in line_stats.line_profiles.values()
                for line_profile in line_profiles
            }
            assert _without_zeros(_v_) == expected_cumulative_times

            # block hit counts
            _v_ = line_stats.block_profiles.items()
            _v_ = {name: block_profile.hit_count for (_, _, name), block_profile in _v_}
            assert _without_zeros(_v_) == _without_zeros(expected._block_hit_counts)

            # block primitive hit counts
            expected_hit_counts = _without_zeros(expected._block_primitive_hit_counts)

            _v_ = {
                name: block_profile.primitive_hit_count
                for (_, _, name), block_profile in line_stats.block_profiles.items()
            }
            assert _without_zeros(_v_) == expected_hit_counts

            # block cumulative times
            _v_ = expected._block_cumulative_times.items()
            _v_ = {name: time * unit_seconds for name, time in _v_}
            expected_cumulative_times = _v_

            _v_ = {
                name: _Time(
                    block_profile.cumulative_time * line_stats.unit, unit_seconds
                )
                for (_, _, name), block_profile in line_stats.block_profiles.items()
            }
            assert _without_zeros(_v_) == expected_cumulative_times

        except Exception as _exception:
            exception = _exception

        else:
            break

    else:
        raise exception


def _sleep(seconds):
    start_count = time.perf_counter_ns()
    end_count = start_count + seconds * 1_000_000_000
    while time.perf_counter_ns() < end_count:
        pass


class _Time:
    def __init__(self, seconds, unit_seconds):
        super().__init__()

        self.seconds = seconds
        self.unit_seconds = unit_seconds

    def __eq__(self, seconds):
        return seconds <= self.seconds and self.seconds <= seconds + self.unit_seconds

    def __repr__(self):
        return f"_Time({repr(self.seconds)}, {repr(self.unit_seconds)})"


def _without_zeros(dictionary):
    return {object: number for object, number in dictionary.items() if number != 0}

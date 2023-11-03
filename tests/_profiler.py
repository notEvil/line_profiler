import asyncio
import contextvars
import inspect
import sys
import types
import typing


_FRAME = lambda object: typing.cast(types.FrameType, object)


class LineProfiler:
    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        self._codes = []
        self._line_hit_counts = {}
        self._line_primitive_hit_counts = {}
        self._line_total_times = {}
        self._line_cumulative_times = {}
        self._block_hit_counts = {}
        self._block_primitive_hit_counts = {}
        self._block_cumulative_times = {}

        self._line12 = contextvars.ContextVar("line12")

        if sys.version_info.minor < 12:
            self.line12 = lambda _: None

    def profile(self, function):
        self._codes.append(function.__code__)

        def _call():
            self._add(self._block_hit_counts, function.__code__.co_name, 1)

            # primitive block
            _v_ = self._get_profiled_frames(self._get_caller_frame().f_back)
            for profiled_frame in _v_:
                if _FRAME(profiled_frame).f_code == function.__code__:
                    break

            else:
                _v_ = function.__code__.co_name
                self._add(self._block_primitive_hit_counts, _v_, 1)

        if inspect.iscoroutinefunction(function):

            async def _function(  # pyright: ignore [reportGeneralTypeIssues]
                *args, **kwargs
            ):
                _call()
                return await function(*args, **kwargs)

        else:

            def _function(*args, **kwargs):
                _call()
                return function(*args, **kwargs)

        return _function

    def line(self, object=None, *_, frame=None):
        if frame is None:
            frame = self._get_caller_frame()

        if self._is_profiled(frame):
            self._add(self._line_hit_counts, frame.f_lineno, 1)

            # primitive line
            for profiled_frame in self._get_profiled_frames(frame.f_back):
                if _FRAME(profiled_frame).f_lineno == frame.f_lineno:
                    break

            else:
                self._add(self._line_primitive_hit_counts, frame.f_lineno, 1)

        self._line12.set(False)
        return object

    def line12(self, _):
        if self._line12.get(False):
            return

        self.line(frame=self._get_caller_frame())
        self._line12.set(True)

    async def async_work(self, _):
        await asyncio.sleep(0)

    def work(self, time):
        profiled_frames = iter(self._get_profiled_frames(self._get_caller_frame()))

        profiled_frame = next(profiled_frames, None)
        if profiled_frame is None:
            return

        self._add(self._line_total_times, profiled_frame.f_lineno, time)
        self._add(self._line_cumulative_times, profiled_frame.f_lineno, time)
        self._add(self._block_cumulative_times, profiled_frame.f_code.co_name, time)

        line_numbers = [profiled_frame.f_lineno]
        codes = [profiled_frame.f_code]
        for profiled_frame in profiled_frames:
            profiled_frame = _FRAME(profiled_frame)

            if profiled_frame.f_lineno not in line_numbers:
                line_numbers.append(profiled_frame.f_lineno)
                self._add(self._line_cumulative_times, profiled_frame.f_lineno, time)

            if profiled_frame.f_code not in codes:
                codes.append(profiled_frame.f_code)

                _v_ = profiled_frame.f_code.co_name
                self._add(self._block_cumulative_times, _v_, time)

    def _add(self, dictionary, key, number):
        if self.debug:
            string = {id(self._line_cumulative_times): "cum"}.get(id(dictionary))
            if string is not None:
                print(f"{key:3} {string} {number}")

        dictionary[key] = dictionary.get(key, 0) + number

    def _get_caller_frame(self) -> types.FrameType:
        return _FRAME(_FRAME(_FRAME(inspect.currentframe()).f_back).f_back)

    def _get_profiled_frames(self, frame):
        while True:
            if self._is_profiled(frame):
                yield frame

            frame = frame.f_back

            if frame is None:
                break

    def _is_profiled(self, frame):
        return frame.f_code in self._codes

# cython: language_level=3
# cython: infer_types=True
# cython: legacy_implicit_noexcept=True
# distutils: language=c++
# distutils: include_dirs = python25.pxd
r"""
This is the Cython backend used in :py:mod:`line_profiler.line_profiler`.

Ignore:
    # Standalone compile instructions for developers
    # Assuming the cwd is the repo root.
    cythonize --annotate --inplace \
        ./line_profiler/_line_profiler.pyx \
        ./line_profiler/timers.c \
        ./line_profiler/unset_trace.c
"""
from .python25 cimport PyFrameObject, PyObject, PyStringObject
from sys import byteorder
import sys
cimport cython
from cpython.version cimport PY_VERSION_HEX
from libc.stdint cimport int64_t

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
import threading

# long long int is at least 64 bytes assuming c99
ctypedef unsigned long long int uint64
ctypedef long long int int64

# FIXME: there might be something special we have to do here for Python 3.11
cdef extern from "frameobject.h":
    """
    inline PyObject* get_frame_code(PyFrameObject* frame) {
        #if PY_VERSION_HEX < 0x030B0000
            Py_INCREF(frame->f_code);
            return frame->f_code;
        #else
            return (PyObject*)PyFrame_GetCode(frame);
        #endif
    }

    inline PyObject* get_code_code(PyCodeObject* code) {
        #if PY_VERSION_HEX < 0x030B0000
            Py_INCREF(code->co_code);
            return code->co_code;
        #else
            return (PyObject*)PyCode_GetCode(code);
        #endif
    }

    inline int get_frame_lineno(PyFrameObject* frame) {
        #if PY_VERSION_HEX < 0x030B0000
            return frame->f_lineno;
        #else
            return PyFrame_GetLineNumber(frame);
        #endif
    }
    """
    cdef object get_frame_code(PyFrameObject* frame)
    cdef object get_code_code(PyCodeObject* code)
    cdef int get_frame_lineno(PyFrameObject* frame)
    ctypedef int (*Py_tracefunc)(object self, PyFrameObject *py_frame, int what, PyObject *arg)

cdef extern from "Python.h":
    """
    // CPython 3.11 broke some stuff by moving PyFrameObject :(
    #if PY_VERSION_HEX >= 0x030b00a6
      #ifndef Py_BUILD_CORE
        #define Py_BUILD_CORE 1
      #endif
      #include "internal/pycore_frame.h"
      #include "cpython/code.h"
      #include "pyframe.h"
    #endif
    """
    ctypedef struct PyFrameObject
    ctypedef struct PyCodeObject
    ctypedef long long PY_LONG_LONG
    cdef bint PyCFunction_Check(object obj)
    cdef int PyCode_Addr2Line(PyCodeObject *co, int byte_offset)

    cdef void PyEval_SetProfile(Py_tracefunc func, object arg)
    cdef void PyEval_SetTrace(Py_tracefunc func, object arg)

    ctypedef object (*PyCFunction)(object self, object args)

    ctypedef struct PyMethodDef:
        char *ml_name
        PyCFunction ml_meth
        int ml_flags
        char *ml_doc

    ctypedef struct PyCFunctionObject:
        PyMethodDef *m_ml
        PyObject *m_self
        PyObject *m_module

    # They're actually #defines, but whatever.
    cdef int PyTrace_CALL
    cdef int PyTrace_EXCEPTION
    cdef int PyTrace_LINE
    cdef int PyTrace_RETURN
    cdef int PyTrace_C_CALL
    cdef int PyTrace_C_EXCEPTION
    cdef int PyTrace_C_RETURN


cdef extern from "timers.c":
    PY_LONG_LONG hpTimer()
    double hpTimerUnit()

cdef extern from "unset_trace.c":
    void unset_trace()

cdef struct CLineProfile:
    int64 code
    int line_number
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG total_time
    PY_LONG_LONG cumulative_time

cdef inline int64 compute_line_hash(uint64 block_hash, uint64 linenum):
    """
    Compute the hash used to store each line timing in an unordered_map.
    This is fairly simple, and could use some improvement since linenum
    isn't technically random, however it seems to be good enough and
    fast enough for any practical purposes.
    """
    # linenum doesn't need to be int64 but it's really a temporary value
    # so it doesn't matter
    return block_hash ^ linenum

def label(code):
    """
    Return a (filename, first_lineno, func_name) tuple for a given code object.

    This is the same labelling as used by the cProfile module in Python 2.5.
    """
    if isinstance(code, str):
        return ("~", 0, code)    # built-in functions ('~' sorts at the end)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)


cpdef _code_replace(func, co_code):
    """
    Implements CodeType.replace for Python < 3.8
    """
    try:
        code = func.__code__
    except AttributeError:
        code = func.__func__.__code__
    if hasattr(code, "replace"):
        # python 3.8+
        code = code.replace(co_code=co_code)
    else:
        # python <3.8
        co = code
        code = type(code)(co.co_argcount, co.co_kwonlyargcount,
                        co.co_nlocals, co.co_stacksize, co.co_flags,
                        co_code, co.co_consts, co.co_names,
                        co.co_varnames, co.co_filename, co.co_name,
                        co.co_firstlineno, co.co_lnotab, co.co_freevars,
                        co.co_cellvars)
    return code


# Note: this is a regular Python class to allow easy pickling.
class LineStats(object):
    """
    Object to encapsulate line-profile statistics.

    Attributes:

        timings (dict):
            Mapping from (filename, first_lineno, function_name) of the
            profiled function to a list of (lineno, nhits, total_time) tuples
            for each profiled line. total_time is an integer in the native
            units of the timer.

        unit (float):
            The number of seconds per timer unit.
    """
    def __init__(self, timings=None, unit=None,
                       line_profiles=None, block_profiles=None, call_profiles=None,
                       call_stats=None):
        super().__init__()

        # backwards compat
        if line_profiles is None:
            line_profiles = timings
        else:
            timings = line_profiles

        self.timings = timings
        self.unit = unit
        self.line_profiles = timings
        self.block_profiles = block_profiles
        self.call_profiles = call_profiles
        self.call_stats = call_stats


class LineProfile(tuple):
    def __init__(self, line_number, hit_count, primitive_hit_count, total_time, cumulative_time):
        super().__init__()

        self.line_number = line_number
        self.hit_count = hit_count
        self.primitive_hit_count = primitive_hit_count
        self.total_time = total_time
        self.cumulative_time = cumulative_time

    # compatibility
    def __new__(cls, line_number, hit_count, primitive_hit_count, total_time, cumulative_time):
        return super().__new__(cls, (line_number, hit_count, cumulative_time))

    def __reduce__(self):
        return (LineProfile, (self.line_number, self.hit_count, self.primitive_hit_count, self.total_time, self.cumulative_time))


class BlockProfile:
    def __init__(self, hit_count, primitive_hit_count, cumulative_time):
        super().__init__()

        self.hit_count = hit_count
        self.primitive_hit_count = primitive_hit_count
        self.cumulative_time = cumulative_time


class CallProfile:
    def __init__(self, hit_count, primitive_hit_count, cumulative_time):
        super().__init__()

        self.hit_count = hit_count
        self.primitive_hit_count = primitive_hit_count
        self.cumulative_time = cumulative_time


class CallStats:
    def __init__(self, total_stats, cumulative_stats):
        super().__init__()

        self.total_stats = total_stats
        self.cumulative_stats = cumulative_stats


class TimeStats:
    def __init__(self, sum_0, sum_1, sum_2, min, max):
        super().__init__()

        self.sum_0 = sum_0
        self.sum_1 = sum_1
        self.sum_2 = sum_2
        self.min = min
        self.max = max


cdef struct Sub:
    PY_LONG_LONG time
    int64 block_hash
    vector[int64] block_hashes
    int line_number
    vector[int] line_numbers
    bint line_hit
    long block_hit
    PY_LONG_LONG total_time
    vector[PY_LONG_LONG] total_times
    PY_LONG_LONG cumulative_time
    vector[PY_LONG_LONG] cumulative_times
    PY_LONG_LONG sub_cumulative_time


cdef struct CBlockProfile:
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG cumulative_time


cdef struct CCallProfile:
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG cumulative_time


cdef struct CCallStats:
    int call_line
    int return_line
    PY_LONG_LONG total_s0
    PY_LONG_LONG total_s1
    double total_s2
    PY_LONG_LONG total_min
    PY_LONG_LONG total_max
    PY_LONG_LONG cumulative_s0
    PY_LONG_LONG cumulative_s1
    double cumulative_s2
    PY_LONG_LONG cumulative_min
    PY_LONG_LONG cumulative_max


cdef class LineProfiler:
    """
    Time the execution of lines of Python code.

    This is the Cython base class for
    :class:`line_profiler.line_profiler.LineProfiler`.

    Example:
        >>> import copy
        >>> import line_profiler
        >>> # Create a LineProfiler instance
        >>> self = line_profiler.LineProfiler()
        >>> # Wrap a function
        >>> copy_fn = self(copy.copy)
        >>> # Call the function
        >>> copy_fn(self)
        >>> # Inspect internal properties
        >>> self.functions
        >>> self.c_last_time
        >>> self.c_code_map
        >>> self.code_map
        >>> self.last_time
        >>> # Print stats
        >>> self.print_stats()
    """
    cdef unordered_map[int64, Sub] _c_subs  # {thread id: Sub}
    cdef unordered_map[int64, unordered_map[int64, CLineProfile]] _c_line_profiles  # {thread id: {code hash: CLineProfile}}
    cdef unordered_map[int64, CBlockProfile] _c_block_profiles  # {block hash: CBlockProfile}
    cdef unordered_map[int64, unordered_map[int64, CCallProfile]] _c_call_profiles  # {block hash: {block hash: CCallProfile}}
    cdef unordered_map[int64, unordered_map[int64, CCallStats]] _c_call_stats  # {code hash: {code hash: CCallStats}}
    cdef public list functions
    cdef public dict block_hash_map, code_hash_map, dupes_map
    cdef public double timer_unit
    cdef public object threaddata

    def __init__(self, *functions):
        self.functions = []
        self.block_hash_map = {}
        self.code_hash_map = {}
        self.dupes_map = {}
        self.timer_unit = hpTimerUnit()
        self.threaddata = threading.local()

        for func in functions:
            self.add_function(func)

    cpdef add_function(self, func):
        """ Record line profiling information for the given Python function.
        """
        if hasattr(func, "__wrapped__"):
            import warnings
            warnings.warn(
                "Adding a function with a __wrapped__ attribute. You may want "
                "to profile the wrapped function by adding %s.__wrapped__ "
                "instead." % (func.__name__,)
            )
        try:
            code = func.__code__
        except AttributeError:
            try:
                code = func.__func__.__code__
            except AttributeError:
                import warnings
                warnings.warn("Could not extract a code object for the object %r" % (func,))
                return

        codes = self.dupes_map.get(code.co_code)
        if codes is None:
            self.dupes_map[code.co_code] = [code]
        else:
            codes.append(code)
            # code hash already exists, so there must be a duplicate function. add no-op
            # co_code = code.co_code + (9).to_bytes(1, byteorder=byteorder) * (len(self.dupes_map[code.co_code]))

            """
            # Code to lookup the NOP opcode, which we will just hard code here
            # instead of looking it up. Perhaps do a global lookup in the
            # future.
            NOP_VALUE: int = opcode.opmap['NOP']
            """
            NOP_VALUE: int = 9
            # Op code should be 2 bytes as stated in
            # https://docs.python.org/3/library/dis.html
            # if sys.version_info[0:2] >= (3, 11):
            NOP_BYTES = NOP_VALUE.to_bytes(2, byteorder=byteorder)
            # else:
            #     NOP_BYTES = NOP_VALUE.to_bytes(1, byteorder=byteorder)

            co_padding = NOP_BYTES * (len(self.dupes_map[code.co_code]) + 1)
            co_code = code.co_code + co_padding
            CodeType = type(code)
            code = _code_replace(func, co_code=co_code)
            try:
                func.__code__ = code
            except AttributeError as e:
                func.__func__.__code__ = code
        block_hash = hash(code.co_code)
        self._c_block_profiles[block_hash]
        self.block_hash_map.setdefault(code, []).append(block_hash)
        # TODO: Since each line can be many bytecodes, this is kinda inefficient
        # See if this can be sped up by not needing to iterate over every byte
        code_hashes = set()
        for offset, byte in enumerate(code.co_code):
            code_hash = compute_line_hash(block_hash, PyCode_Addr2Line(<PyCodeObject*>code, offset))
            code_hashes.add(code_hash)
            self._c_line_profiles[code_hash]
        self.code_hash_map[code] = code_hashes

        self.functions.append(func)

    property enable_count:
        def __get__(self):
            if not hasattr(self.threaddata, "enable_count"):
                self.threaddata.enable_count = 0
            return self.threaddata.enable_count
        def __set__(self, value):
            self.threaddata.enable_count = value

    def enable_by_count(self):
        """ Enable the profiler if it hasn't been enabled before.
        """
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        PyEval_SetTrace(python_trace_callback, self)

    @property
    def c_code_map(self):
        """
        A Python view of the internal C lookup table.
        """
        return <dict>self._c_line_profiles

    @property
    def c_last_time(self):
        """
        TODO for backwards compatibility
        """
        cdef Sub* sub = &self._c_subs[threading.get_ident()]
        return {sub.block_hash: dict(f_lineno=sub.line_number, time=sub.time)}

    @property
    def code_map(self):
        """
        line_profiler 4.0 no longer directly maintains code_map, but this will
        construct something similar for backwards compatibility.
        """
        cdef dict line_profiles = self._c_line_profiles
        code_map = {}
        for code, code_hashes in self.code_hash_map.items():
            entries = code_map.setdefault(code, {})
            for code_hash in code_hashes:
                for key, entry in line_profiles[code_hash].items():
                    entry = entry.copy()
                    entry["code"] = code
                    entries[key] = entry
        return code_map

    @property
    def last_time(self):
        """
        line_profiler 4.0 no longer directly maintains last_time, but this will
        construct something similar for backwards compatibility.
        """
        cdef Sub* sub = &self._c_subs[threading.get_ident()]
        code_hash = compute_line_hash(sub.block_hash, sub.line_number)
        return {
            code: dict(f_lineno=sub.line_number, time=sub.time)
            for code, code_hashes in self.code_hash_map.items()
            if code_hash in code_hashes
        }

    cpdef disable(self):
        unset_trace()

    def get_stats(self):
        """
        Return a LineStats object containing the timings.
        """
        codes = {
            code_hash: code
            for code, code_hashes in self.code_hash_map.items()
            for code_hash in code_hashes
        }
        codes.update(
            (block_hash, code)
            for code, block_hashes in self.block_hash_map.items()
            for block_hash in block_hashes
        )
        
        c_line_profiles = <dict>self._c_line_profiles
        line_profiles = {}
        for code, code_hashes in self.code_hash_map.items():
            numbers = {}
            for code_hash in code_hashes:
                for entry in c_line_profiles[code_hash].values():
                    line_number = entry["line_number"]
                    hit_count, primitive_hit_count, total_time, cumulative_time = numbers.get(line_number, (0, 0, 0, 0))
                    numbers[line_number] = (
                        hit_count + entry["hit_count"],
                        primitive_hit_count + entry["primitive_hit_count"],
                        total_time + entry["total_time"],
                        cumulative_time + entry["cumulative_time"],
                    )
            line_profiles[label(code)] = [
                LineProfile(line_number, hit_count, primitive_hit_count, total_time, cumulative_time)
                for line_number, (hit_count, primitive_hit_count, total_time, cumulative_time) in numbers.items()
            ]

        block_profiles = {
            label(codes[block_hash]): BlockProfile(dictionary["hit_count"], dictionary["primitive_hit_count"], dictionary["cumulative_time"])
            for block_hash, dictionary in (<dict>self._c_block_profiles).items()
        }

        call_profiles = {}
        for caller_hash, dictionary in (<dict>self._c_call_profiles).items():
            caller_profiles = call_profiles.setdefault(None if caller_hash == 0 else label(codes[caller_hash]), {})
            for callee_hash, dictionary in dictionary.items():
                caller_profiles[label(codes[callee_hash])] = CallProfile(dictionary["hit_count"], dictionary["primitive_hit_count"], dictionary["cumulative_time"])

        call_stats = {}
        for caller_hash, dictionary in (<dict>self._c_call_stats).items():
            caller_stats = call_stats.setdefault(None if caller_hash == 0 else label(codes[caller_hash]), {})
            for callee_hash, dictionary in dictionary.items():
                callee_stats = caller_stats.setdefault(None if caller_hash == 0 else dictionary["call_line"], {})
                callee_stats = callee_stats.setdefault(label(codes[callee_hash]), {})
                callee_stats[dictionary["return_line"]] = CallStats(
                    total_stats=TimeStats(
                        sum_0=dictionary["total_s0"], sum_1=dictionary["total_s1"], sum_2=dictionary["total_s2"],
                        min=dictionary["total_min"], max=dictionary["total_max"],
                    ),
                    cumulative_stats=TimeStats(
                        sum_0=dictionary["cumulative_s0"], sum_1=dictionary["cumulative_s1"], sum_2=dictionary["cumulative_s2"],
                        min=dictionary["cumulative_min"], max=dictionary["cumulative_max"],
                    ),
                )

        return LineStats(line_profiles=line_profiles, block_profiles=block_profiles, call_profiles=call_profiles,
                         call_stats=call_stats, unit=self.timer_unit)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef extern int python_trace_callback(object self_, PyFrameObject *py_frame,
                                      int what, PyObject *arg):
    """
    The PyEval_SetTrace() callback.

    References:
       https://github.com/python/cpython/blob/de2a4036/Include/cpython/pystate.h#L16 
    """
    cdef LineProfiler self
    cdef PY_LONG_LONG time
    cdef object code
    cdef int64 block_hash
    cdef Sub* sub
    cdef int line_number
    cdef int64 code_hash
    cdef int index
    cdef CCallProfile* call_profile
    cdef CCallStats* call_stats

    self = <LineProfiler>self_

    if what == PyTrace_LINE or what == PyTrace_CALL or what == PyTrace_RETURN:
        """
        - sequence: CALL -> (?P<sub>(LINE | (CALL -> (?P=sub)* -> RETURN)) -> )* -> RETURN
          - `yield from` creates CALL -> CALL -> ... -> RETURN -> RETURN
          - multiple calls on a single line create CALL -> ... -> RETURN -> CALL -> ... -> RETURN
        """
        time = hpTimer()
        # Normally we'd need to DECREF the return from get_frame_code and get_code_code, but Cython does that for us
        code = get_frame_code(py_frame)
        block_hash = hash(get_code_code(<PyCodeObject*>code))
        if self._c_block_profiles.count(block_hash):
            ident = threading.get_ident()
            sub = &self._c_subs[ident]

            line_number = get_frame_lineno(py_frame)
            if line_number == -1:
                # assert block_hash == sub.block_hash
                line_number = sub.line_number

            code_hash = compute_line_hash(block_hash, line_number)

            _record(self, time, sub)  # count hit and attribute time to the previous line/block

            if what == PyTrace_CALL:
                if line_number == code.co_firstlineno:  # function call or start generator or coroutine
                    sub.block_hit = 0

                else:  # continue generator or coroutine
                    _enter_call(sub)
                    sub.block_hash = block_hash
                    sub.line_number = line_number

            elif what == PyTrace_LINE:
                if sub.block_hit == 1:
                    _enter_call(sub)

                sub.block_hash = block_hash
                sub.line_number = line_number
                sub.line_hit = True

            elif what == PyTrace_RETURN:
                call_profile = &self._c_call_profiles[sub.block_hashes.back()][block_hash]

                call_profile.hit_count += 1

                # primitive call
                for index in range(sub.block_hashes.size() - 1):
                    if sub.block_hashes[index] == sub.block_hashes.back() \
                       and sub.block_hashes[index + 1] == block_hash:
                        break
                else:
                    call_profile.primitive_hit_count += 1
                    call_profile.cumulative_time += sub.cumulative_time

                # pop location from stack
                sub.block_hash = sub.block_hashes.back()
                sub.block_hashes.pop_back()
                sub.line_number = sub.line_numbers.back()
                sub.line_numbers.pop_back()
                #

                call_stats = &self._c_call_stats[compute_line_hash(sub.block_hash, sub.line_number)][code_hash]
                if call_stats.total_s0 == 0:
                    call_stats.call_line = sub.line_number
                    call_stats.return_line = line_number
                    call_stats.cumulative_min = sub.cumulative_time
                    call_stats.cumulative_max = sub.cumulative_time
                    call_stats.total_min = sub.total_time
                    call_stats.total_max = sub.total_time

                call_stats.total_s0 += 1
                call_stats.total_s1 += sub.total_time
                call_stats.total_s2 += <double>sub.total_time * <double>sub.total_time
                call_stats.total_min = min(call_stats.total_min, sub.total_time)
                call_stats.total_max = max(call_stats.total_max, sub.total_time)
                call_stats.cumulative_s0 += 1
                call_stats.cumulative_s1 += sub.cumulative_time
                call_stats.cumulative_s2 += <double>sub.cumulative_time * <double>sub.cumulative_time
                call_stats.cumulative_min = min(call_stats.cumulative_min, sub.cumulative_time)
                call_stats.cumulative_max = max(call_stats.cumulative_max, sub.cumulative_time)

                sub.sub_cumulative_time = sub.cumulative_time  # similar to line_hit and block_hit

                # pop remaining from stack
                sub.total_time = sub.total_times.back()
                sub.total_times.pop_back()
                sub.cumulative_time = sub.cumulative_times.back()
                sub.cumulative_times.pop_back()

            sub.time = hpTimer()

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _record(LineProfiler self, PY_LONG_LONG time, Sub* sub):
    cdef int64 code_hash
    cdef CLineProfile* line_profile
    cdef CBlockProfile* block_profile
    cdef int index

    if not (sub.block_hash == 0):  # not (just entering)
        code_hash = compute_line_hash(sub.block_hash, sub.line_number)

        line_profile = &self._c_line_profiles[code_hash][sub.line_number]  # get or create CLineProfile
        line_profile.code = code_hash
        line_profile.line_number = sub.line_number

        line_profile.hit_count += sub.line_hit
        line_profile.total_time += time - sub.time

        # primitive line
        index = 0
        for block_hash in sub.block_hashes:
            if compute_line_hash(block_hash, sub.line_numbers[index]) == code_hash:
                break
            index += 1
        else:
            line_profile.primitive_hit_count += sub.line_hit
            line_profile.cumulative_time += time - sub.time + sub.sub_cumulative_time
        #

        if sub.block_hit == 1:  # after first LINE after function call
            block_profile = &self._c_block_profiles[sub.block_hash]
            block_profile.hit_count += 1

        # primitive block
        for block_hash in sub.block_hashes:
            if block_hash == sub.block_hash:
                break
        else:
            block_profile = &self._c_block_profiles[sub.block_hash]
            block_profile.primitive_hit_count += sub.block_hit == 1
            block_profile.cumulative_time += time - sub.time + sub.sub_cumulative_time
        #

        sub.total_time += time - sub.time
        sub.cumulative_time += time - sub.time + sub.sub_cumulative_time

        # reset
        sub.line_hit = False
        sub.sub_cumulative_time = 0

    sub.block_hit += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _enter_call(Sub* sub):
    sub.block_hashes.push_back(sub.block_hash)
    sub.line_numbers.push_back(sub.line_number)
    sub.total_times.push_back(sub.total_time)
    sub.cumulative_times.push_back(sub.cumulative_time)

    sub.total_time = 0
    sub.cumulative_time = 0
    sub.sub_cumulative_time = 0

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

from parallel_hashmap cimport flat_hash_map, parallel_flat_hash_map
from preshed.maps cimport PreshMap

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

    ctypedef struct Py_tss_t
    cdef Py_tss_t Py_tss_NEEDS_INIT
    cdef int PyThread_tss_create(Py_tss_t *key)
    cdef void *PyThread_tss_get(Py_tss_t *key)
    cdef int PyThread_tss_set(Py_tss_t *key, void *value)


cdef extern from "timers.c":
    PY_LONG_LONG hpTimer()
    double hpTimerUnit()

cdef extern from "unset_trace.c":
    void unset_trace()

cdef struct CLineProfile:
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG total_time
    PY_LONG_LONG cumulative_time

cdef int LINE_BITS = 24
cdef uint64 LINE_MASK = ((<uint64>1) << LINE_BITS) - 1
cdef int BLOCK_BITS = 64 - LINE_BITS
cdef uint64 BLOCK_MASK = ((<uint64>1) << BLOCK_BITS) - 1

cdef inline uint64 get_line_id(uint64 block_id, int line_number):
    """
    Compute the hash used to store each line timing in an unordered_map.
    This is fairly simple, and could use some improvement since linenum
    isn't technically random, however it seems to be good enough and
    fast enough for any practical purposes.
    """
    # linenum doesn't need to be int64 but it's really a temporary value
    # so it doesn't matter
    return (block_id << LINE_BITS) | ((block_id + line_number) & LINE_MASK)

cdef inline uint64 get_code_block_id(PyObject* code_bytes):
    return (<uint64>code_bytes) & BLOCK_MASK
    # return hash(<object>code_bytes) & BLOCK_MASK

cdef inline uint64 get_line_block_id(uint64 line_id):
    return line_id >> LINE_BITS

cdef inline int get_line_number(uint64 line_id):
    cdef int line_number = (<int>(line_id & LINE_MASK)) - (<int>(get_line_block_id(line_id) & LINE_MASK))
    return (line_number + (1 << LINE_BITS)) if line_number < 0 else line_number

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
    uint64 line_id
    bint line_hit
    long block_hit
    PY_LONG_LONG total_time
    PY_LONG_LONG cumulative_time
    PY_LONG_LONG sub_cumulative_time
    vector[uint64] line_ids
    vector[PY_LONG_LONG] total_times
    vector[PY_LONG_LONG] cumulative_times


cdef struct CBlockProfile:
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG cumulative_time


cdef struct CCallProfile:
    long hit_count
    long primitive_hit_count
    PY_LONG_LONG cumulative_time


cdef struct CCallStats:
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


cdef Py_tss_t tss_key = Py_tss_NEEDS_INIT

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
    cdef parallel_flat_hash_map[int64, Sub] _c_subs  # {thread id: Sub}
    cdef parallel_flat_hash_map[uint64, CLineProfile] _c_line_profiles  # {line id: CLineProfile}
    cdef parallel_flat_hash_map[uint64, CBlockProfile] _c_block_profiles  # {block id: CBlockProfile}
    cdef parallel_flat_hash_map[uint64, flat_hash_map[uint64, CCallProfile]] _c_call_profiles  # {call block id: {return block id: CCallProfile}}
    cdef parallel_flat_hash_map[uint64, flat_hash_map[uint64, CCallStats]] _c_call_stats  # {call line id: {return line id: CCallStats}}
    cdef public list functions
    cdef public dict block_map, dupes_map
    cdef public double timer_unit
    cdef public object threaddata
    cdef PreshMap presh_map

    def __init__(self, *functions):
        self.functions = []
        self.block_map = {}
        self.dupes_map = {}
        self.timer_unit = hpTimerUnit()
        self.threaddata = threading.local()

        for func in functions:
            self.add_function(func)

        self.presh_map = PreshMap(256)

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
        code_bytes = get_code_code(<PyCodeObject*>code)
        block_id = get_code_block_id(<PyObject*>code_bytes)
        self._c_block_profiles[block_id]
        self.block_map[block_id] = code
        self.presh_map.set(block_id, <void*>1)
        # TODO: Since each line can be many bytecodes, this is kinda inefficient
        # See if this can be sped up by not needing to iterate over every byte

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
        raise NotImplementedError

    @property
    def c_last_time(self):
        raise NotImplementedError

    @property
    def code_map(self):
        """
        line_profiler 4.0 no longer directly maintains code_map, but this will
        construct something similar for backwards compatibility.
        """
        raise NotImplementedError

    @property
    def last_time(self):
        """
        line_profiler 4.0 no longer directly maintains last_time, but this will
        construct something similar for backwards compatibility.
        """
        raise NotImplementedError

    cpdef disable(self):
        unset_trace()

    def get_stats(self):
        """
        Return a LineStats object containing the timings.
        """
        
        line_profiles = {}
        for line_id, dictionary in ({item.first: <dict>item.second for item in self._c_line_profiles}).items():
            key = label(self.block_map[get_line_block_id(line_id)])
            line_profiles.setdefault(key, []).append(LineProfile(
                get_line_number(line_id),
                dictionary["hit_count"], dictionary["primitive_hit_count"],
                dictionary["total_time"], dictionary["cumulative_time"],
            ))

        block_profiles = {
            label(self.block_map[block_id]): BlockProfile(
                dictionary["hit_count"], dictionary["primitive_hit_count"],
                dictionary["cumulative_time"],
            )
            for block_id, dictionary in ({item.first: <dict>item.second for item in self._c_block_profiles}).items()
        }

        call_profiles = {
            None if call_block_id == 0 else label(self.block_map[call_block_id]): {
                label(self.block_map[return_block_id]): CallProfile(
                    dictionary["hit_count"], dictionary["primitive_hit_count"],
                    dictionary["cumulative_time"],
                )
                for return_block_id, dictionary in _dictionary.items()
            }
            for call_block_id, _dictionary in ({item.first: {item.first: <dict>item.second for item in item.second} for item in self._c_call_profiles}).items()
        }

        call_stats = {}
        for call_line_id, dictionary in ({item.first: {item.first: <dict>item.second for item in item.second} for item in self._c_call_stats}).items():
            key = None if call_line_id == 0 else label(self.block_map[get_line_block_id(call_line_id)])
            _call_stats = call_stats.setdefault(key, {})
            key = None if call_line_id == 0 else get_line_number(call_line_id)
            _call_stats = _call_stats.setdefault(key, {})
            for return_line_id, dictionary in dictionary.items():
                key = label(self.block_map[get_line_block_id(return_line_id)])
                _call_stats.setdefault(key, {})[get_line_number(return_line_id)] = CallStats(
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


_threading_get_ident = threading.get_ident

cdef int python_trace_callback(object self_, PyFrameObject *py_frame, int what,
PyObject *arg):
    """
    The PyEval_SetTrace() callback.

    References:
       https://github.com/python/cpython/blob/de2a4036/Include/cpython/pystate.h#L16 
    """
    cdef LineProfiler self
    cdef PY_LONG_LONG time
    cdef object code
    cdef object code_bytes
    cdef uint64 block_id
    cdef int64 thread_id
    cdef Sub* sub
    cdef int line_number
    cdef CCallProfile* call_profile
    cdef int index
    cdef CCallStats* call_stats
    cdef bint inner_time = True

    self = <LineProfiler>self_

    if what == PyTrace_LINE or what == PyTrace_CALL or what == PyTrace_RETURN:
        """
        - sequence: CALL -> (?P<sub>(LINE | (CALL -> (?P=sub)* -> RETURN)) -> )* -> RETURN
          - `yield from` creates CALL -> CALL -> ... -> RETURN -> RETURN
          - multiple calls on a single line create CALL -> ... -> RETURN -> CALL -> ... -> RETURN
        """
        if not inner_time:
            time = hpTimer()
        # Normally we'd need to DECREF the return from get_frame_code and get_code_code, but Cython does that for us
        code = get_frame_code(py_frame)
        code_bytes = get_code_code(<PyCodeObject*>code)
        block_id = get_code_block_id(<PyObject*>code_bytes)
        if self.presh_map.get(block_id) if True else self._c_block_profiles.count(block_id):
            if inner_time:
                time = hpTimer()

            if True:
                PyThread_tss_create(&tss_key)
                thread_id = <int64>PyThread_tss_get(&tss_key)
                if thread_id == 0:
                    thread_id = threading.get_ident()
                    PyThread_tss_set(&tss_key, <void*>thread_id)
                # assert thread_id == threading.get_ident()
                sub = &self._c_subs[thread_id]

            else:
                sub = &self._c_subs[_threading_get_ident()]

            line_number = get_frame_lineno(py_frame)
            if line_number == -1:
                # assert block_id == get_line_block_id(sub.line_id)
                line_number = get_line_number(sub.line_id)

            _record(self, time, sub)  # count hit and attribute time to the previous line/block

            if what == PyTrace_CALL:
                if line_number == code.co_firstlineno:  # function call or start generator or coroutine
                    sub.block_hit = 0

                else:  # continue generator or coroutine
                    _enter_call(sub)
                    sub.line_id = get_line_id(block_id, line_number)

            elif what == PyTrace_LINE:
                if sub.block_hit == 1:
                    _enter_call(sub)

                sub.line_id = get_line_id(block_id, line_number)
                sub.line_hit = True

            elif what == PyTrace_RETURN:
                call_profile = &self._c_call_profiles[get_line_block_id(sub.line_ids.back())][block_id]

                call_profile.hit_count += 1

                # primitive call
                for index in range(sub.line_ids.size() - 1):
                    if get_line_block_id(sub.line_ids[index]) == get_line_block_id(sub.line_ids.back()) \
                       and get_line_block_id(sub.line_ids[index + 1]) == block_id:
                        break
                else:
                    call_profile.primitive_hit_count += 1
                    call_profile.cumulative_time += sub.cumulative_time

                # pop location from stack
                sub.line_id = sub.line_ids.back()
                sub.line_ids.pop_back()
                #

                call_stats = &self._c_call_stats[sub.line_id][get_line_id(block_id, line_number)]
                if call_stats.total_s0 == 0:
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


cdef void _record(LineProfiler self, PY_LONG_LONG time, Sub* sub):
    cdef PY_LONG_LONG difference
    cdef PY_LONG_LONG cumulative_difference
    cdef CLineProfile* line_profile
    cdef uint64 line_id
    cdef CBlockProfile* block_profile

    if not (sub.line_id == 0):  # not (just entering)
        difference = time - sub.time
        cumulative_difference = difference + sub.sub_cumulative_time

        line_profile = &self._c_line_profiles[sub.line_id]  # get or create CLineProfile

        line_profile.hit_count += sub.line_hit
        line_profile.total_time += difference

        # primitive line
        for line_id in sub.line_ids:
            if line_id == sub.line_id:
                break
        else:
            line_profile.primitive_hit_count += sub.line_hit
            line_profile.cumulative_time += cumulative_difference
        #

        if sub.block_hit == 1:  # after first LINE after function call
            block_profile = &self._c_block_profiles[get_line_block_id(sub.line_id)]
            block_profile.hit_count += 1

        # primitive block
        for line_id in sub.line_ids:
            if get_line_block_id(line_id) == get_line_block_id(sub.line_id):
                break
        else:
            block_profile = &self._c_block_profiles[get_line_block_id(sub.line_id)]
            block_profile.primitive_hit_count += sub.block_hit == 1
            block_profile.cumulative_time += cumulative_difference
        #

        sub.total_time += difference
        sub.cumulative_time += cumulative_difference

        # reset
        sub.line_hit = False
        sub.sub_cumulative_time = 0

    sub.block_hit += 1


cdef void _enter_call(Sub* sub):
    sub.line_ids.push_back(sub.line_id)
    sub.total_times.push_back(sub.total_time)
    sub.cumulative_times.push_back(sub.cumulative_time)

    sub.total_time = 0
    sub.cumulative_time = 0
    sub.sub_cumulative_time = 0

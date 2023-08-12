#cython: language_level=3
from .python25 cimport PyFrameObject, PyObject, PyStringObject
from sys import byteorder
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
            Py_INCREF(frame->f_code->co_code);
            return frame->f_code->co_code;
        #else
            PyCodeObject* code = PyFrame_GetCode(frame);
            PyObject* ret = PyCode_GetCode(code);
            Py_DECREF(code);
            return ret;
        #endif
    }
    inline int get_line_number(PyFrameObject* frame) {
        #if PY_VERSION_HEX < 0x030B0000
            return frame->f_lineno;
        #else
            return PyFrame_GetLineNumber(frame);
        #endif
    }
    """
    cdef object get_frame_code(PyFrameObject* frame)
    cdef int get_line_number(PyFrameObject* frame)
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

cdef struct LineTime:
    int64 code
    int lineno
    PY_LONG_LONG cumulative_time
    PY_LONG_LONG new_total_time
    PY_LONG_LONG new_cumulative_time
    long nhits
    
cdef struct LastTime:
    int f_lineno
    PY_LONG_LONG time

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
    """ Return a (filename, first_lineno, func_name) tuple for a given code
    object.

    This is the same labelling as used by the cProfile module in Python 2.5.
    """
    if isinstance(code, str):
        return ('~', 0, code)    # built-in functions ('~' sorts at the end)
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
    if hasattr(code, 'replace'):
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
    """ Object to encapsulate line-profile statistics.

    Attributes
    ----------
    timings : dict
        Mapping from (filename, first_lineno, function_name) of the profiled
        function to a list of (lineno, nhits, cumulative_time) tuples for each
        profiled line. cumulative_time is an integer in the native units of the
        timer.
    unit : float
        The number of seconds per timer unit.
    """
    def __init__(self, timings, new_timings, new_call_stats, unit):
        self.timings = timings
        self.new_timings = new_timings
        self.new_call_stats = new_call_stats
        self.unit = unit


cdef struct _Sub:
    PY_LONG_LONG time
    int64 code_hash
    vector[int64] code_hashes
    int line_number
    vector[int] line_numbers
    PY_LONG_LONG total_time
    vector[PY_LONG_LONG] total_times
    PY_LONG_LONG cumulative_time
    vector[PY_LONG_LONG] cumulative_times
    PY_LONG_LONG sub_cumulative_time


cdef struct _Call:
    int call_line
    int return_line
    PY_LONG_LONG total_s0
    PY_LONG_LONG total_s1
    double total_s2
    PY_LONG_LONG total_min_time
    PY_LONG_LONG total_max_time
    PY_LONG_LONG cumulative_s0
    PY_LONG_LONG cumulative_s1
    double cumulative_s2
    PY_LONG_LONG cumulative_min_time
    PY_LONG_LONG cumulative_max_time


cdef class LineProfiler:
    """ 
    Time the execution of lines of Python code.

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
    cdef unordered_map[int64, unordered_map[int64, LineTime]] _c_code_map
    # Mapping between thread-id and map of LastTime
    cdef unordered_map[int64, unordered_map[int64, LastTime]] _c_last_time
    cdef unordered_map[int64, _Sub] _c_sub
    cdef unordered_map[int64, unordered_map[int64, _Call]] _c_call
    cdef public list functions
    cdef public dict code_hash_map, dupes_map
    cdef public double timer_unit
    cdef public object threaddata

    def __init__(self, *functions):
        self.functions = []
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
            co_code = code.co_code + (9).to_bytes(1, byteorder=byteorder) * len(codes)
            CodeType = type(code)
            code = _code_replace(func, co_code=co_code)
            try:
                func.__code__ = code
            except AttributeError as e:
                func.__func__.__code__ = code
        # TODO: Since each line can be many bytecodes, this is kinda inefficient
        # See if this can be sped up by not needing to iterate over every byte
        code_hashes = self.code_hash_map.get(code)
        if code_hashes is None:
            previous_line = -1
            block_hash = hash(code.co_code)
            code_hashes = []
            for offset, byte in enumerate(code.co_code):
                line = PyCode_Addr2Line(<PyCodeObject*>code, offset)
                if line != previous_line:
                    code_hash = compute_line_hash(block_hash, line)
                    code_hashes.append(code_hash)
                    self._c_code_map[code_hash]
                    previous_line = line
            self.code_hash_map[code] = code_hashes
            self._c_code_map[compute_line_hash(block_hash, code.co_firstlineno)]  # PyTrace_CALL

        self.functions.append(func)

    property enable_count:
        def __get__(self):
            if not hasattr(self.threaddata, 'enable_count'):
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
        return <dict>self._c_code_map
        
    @property
    def c_last_time(self):
        return (<dict>self._c_last_time)[threading.get_ident()]

    @property
    def code_map(self):
        """
        line_profiler 4.0 no longer directly maintains code_map, but this will
        construct something similar for backwards compatibility.
        """
        c_code_map = self.c_code_map
        py_code_map = {}
        for code, code_hashes in self.code_hash_map.items():
            py_code_map.setdefault(code, {})
            for code_hash in code_hashes:
                c_entries = c_code_map[code_hash]
                py_entries = {}
                for key, c_entry in c_entries.items():
                    py_entry = c_entry.copy()
                    py_entry['code'] = code
                    py_entries[key] = py_entry
                py_code_map[code].update(py_entries)
        return py_code_map

    @property
    def last_time(self):
        """
        line_profiler 4.0 no longer directly maintains last_time, but this will
        construct something similar for backwards compatibility.
        """
        c_last_time = (<dict>self._c_last_time)[threading.get_ident()]
        py_last_time = {}
        for code, code_hashes in self.code_hash_map.items():
            for code_hash in code_hashes:
                if code_hash in c_last_time:
                    py_last_time[code] = c_last_time[code_hash]
        return py_last_time


    cpdef disable(self):
        self._c_last_time[threading.get_ident()].clear()
        unset_trace()

    def get_stats(self):
        """ Return a LineStats object containing the timings.
        """
        cdef dict cmap
        
        codes = {code_hash: code for code, code_hashes in self.code_hash_map.items() for code_hash in code_hashes}

        stats = {}
        new_timings = {}
        new_call_stats = {}
        for code, code_hashes in self.code_hash_map.items():
            cmap = self._c_code_map
            entries = []
            for code_hash in code_hashes:
                entries.extend(cmap[code_hash].values())
            key = label(code)

            numbers = {}
            for e in entries:
                line_number = e["lineno"]
                nhits, cumulative_time, new_total_time, new_cumulative_time = numbers.get(line_number, (0, 0, 0, 0))
                numbers[line_number] = (
                    nhits + e["nhits"],
                    cumulative_time + e["cumulative_time"],
                    new_total_time + e["new_total_time"],
                    new_cumulative_time + e["new_cumulative_time"],
                )
            entries = [(line_number, nhits, cumulative_time) for line_number, (nhits, cumulative_time, _, _) in numbers.items()]
            new_timings[key] = [(line_number, new_total_time, new_cumulative_time) for line_number, (_, _, new_total_time, new_cumulative_time) in numbers.items()]

            cmap = self._c_call
            for caller_hash, calls in cmap.items():
                caller_stats = new_call_stats.setdefault(None if caller_hash == 0 else label(codes[caller_hash]), {})
                for callee_hash, call in calls.items():
                    callee_stats = caller_stats.setdefault(None if caller_hash == 0 else call['call_line'], {})
                    callee_stats = callee_stats.setdefault(label(codes[callee_hash]), {})
                    callee_stats[call['return_line']] = (
                        (call['total_s0'], call['total_s1'], call['total_s2'], call['total_min_time'], call['total_max_time']),
                        (call['cumulative_s0'], call['cumulative_s1'], call['cumulative_s2'], call['cumulative_min_time'], call['cumulative_max_time']),
                    )

            stats[key] = entries
        return LineStats(stats, new_timings, new_call_stats, self.timer_unit)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int python_trace_callback(object self_, PyFrameObject *py_frame, int what,
PyObject *arg):
    """ The PyEval_SetTrace() callback.
    """
    cdef LineProfiler self
    cdef PY_LONG_LONG time
    cdef int64 block_hash
    cdef int64 code_hash
    cdef _Sub* sub
    cdef _Call* call
    cdef LastTime old

    self = <LineProfiler>self_

    if what == PyTrace_LINE or what == PyTrace_CALL or what == PyTrace_RETURN:
        # Normally we'd need to DECREF the return from get_frame_code, but Cython does that for us
        time = hpTimer()
        block_hash = hash(get_frame_code(py_frame))
        code_hash = compute_line_hash(block_hash, get_line_number(py_frame))
        if self._c_code_map.count(code_hash):
            ident = threading.get_ident()
            sub = &self._c_sub[ident]

            if what == PyTrace_CALL:
                _record_time(self, time, sub)

                sub.code_hashes.push_back(sub.code_hash)
                sub.line_numbers.push_back(sub.line_number)
                sub.total_times.push_back(sub.total_time)
                sub.cumulative_times.push_back(sub.cumulative_time)

                sub.code_hash = 0
                sub.total_time = 0
                sub.cumulative_time = 0
                sub.sub_cumulative_time = 0

            elif what == PyTrace_LINE:
                _record_time(self, time, sub)

                sub.code_hash = code_hash
                sub.line_number = get_line_number(py_frame)
                sub.time = hpTimer()

            elif what == PyTrace_RETURN:
                _record_time(self, time, sub)

                sub.code_hash = sub.code_hashes.back()
                sub.code_hashes.pop_back()
                sub.line_number = sub.line_numbers.back()
                sub.line_numbers.pop_back()

                call = &self._c_call[sub.code_hash][code_hash]
                if call.total_s0 == 0:
                    call.call_line = sub.line_number
                    call.return_line = get_line_number(py_frame)
                    call.cumulative_min_time = sub.cumulative_time
                    call.cumulative_max_time = sub.cumulative_time
                    call.total_min_time = sub.total_time
                    call.total_max_time = sub.total_time
                call.total_s0 += 1
                call.total_s1 += sub.total_time
                call.total_s2 += <double>sub.total_time * <double>sub.total_time
                call.total_min_time = min(call.total_min_time, sub.total_time)
                call.total_max_time = max(call.total_max_time, sub.total_time)
                call.cumulative_s0 += 1
                call.cumulative_s1 += sub.cumulative_time
                call.cumulative_s2 += <double>sub.cumulative_time * <double>sub.cumulative_time
                call.cumulative_min_time = min(call.cumulative_min_time, sub.cumulative_time)
                call.cumulative_max_time = max(call.cumulative_max_time, sub.cumulative_time)

                sub.total_time = sub.total_times.back()
                sub.total_times.pop_back()
                sub.sub_cumulative_time = sub.cumulative_time
                sub.cumulative_time = sub.cumulative_times.back()
                sub.cumulative_times.pop_back()

                sub.time = hpTimer()

            # original
            if self._c_last_time[ident].count(block_hash):
                old = self._c_last_time[ident][block_hash]
                line_entries = self._c_code_map[code_hash]
                key = old.f_lineno
                if not line_entries.count(key):
                    self._c_code_map[code_hash][key] = LineTime(code_hash, key, 0, 0)
                self._c_code_map[code_hash][key].nhits += 1
                self._c_code_map[code_hash][key].cumulative_time += time - old.time
            if what == PyTrace_LINE:
                # Get the time again. This way, we don't record much time wasted
                # in this function.
                self._c_last_time[ident][block_hash] = LastTime(get_line_number(py_frame), hpTimer())
            elif self._c_last_time[ident].count(block_hash):
                # We are returning from a function, not executing a line. Delete
                # the last_time record. It may have already been deleted if we
                # are profiling a generator that is being pumped past its end.
                self._c_last_time[ident].erase(self._c_last_time[ident].find(block_hash))

    return 0


cdef _record_time(LineProfiler self, PY_LONG_LONG time, _Sub* sub):
    if sub.code_hash != 0:
        line_time = &self._c_code_map[sub.code_hash][sub.line_number]
        if line_time.code == 0:
            line_time.code = sub.code_hash
            line_time.lineno = sub.line_number
        line_time.new_total_time += time - sub.time
        line_time.new_cumulative_time += time - sub.time + sub.sub_cumulative_time

        sub.total_time += time - sub.time
        sub.cumulative_time += time - sub.time + sub.sub_cumulative_time

        sub.sub_cumulative_time = 0

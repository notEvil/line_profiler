from subprocess_shell import *
import typer
import os
import pathlib
import sys
import typing


def main(
    path: pathlib.Path,
    pyperformance_path: pathlib.Path,
    line_profiler_path: pathlib.Path,
    python_path: typing.Optional[pathlib.Path] = None,
):
    def _start(*args, cwd=path, env=os.environ | dict(PIPENV_NO_INHERIT="1"), **kwargs):
        return start(*args, cwd=cwd, env=env, **kwargs)

    path.mkdir(parents=True, exist_ok=True)

    if python_path is not None:
        _v_ = [sys.executable, "-m", "pipenv", "--python", python_path] >> _start()
        _ = _v_ >> wait()

    _v_ = [sys.executable, "-m", "pipenv", "run", "pip", "install", pyperformance_path]
    _ = _v_ >> _start() >> wait()

    _v_ = "pyperformance"
    _ = (
        [sys.executable, "-m", "pipenv", "run", "python", "-m", _v_, "venv", "recreate"]
        >> _start(env=os.environ | dict(_LINE_PROFILER=str(line_profiler_path)))
        >> wait()
    )


typer.run(main)

"""Run the existing workers without a Qt event loop."""
import json
import os
import queue
import subprocess
import threading
import time
from contextlib import contextmanager

from .paths import ROOT, worker_command
from .processing_guard import ProcessingGuard
from .queue_groups import BATCH_STAGES, argument, remove_request, write_request

EVENT_PREFIX = '@@MOCAP@@'
RETAINED_EVENTS = {'prepared', 'people_found', 'person_prepared'}


class WorkerError(RuntimeError):
    pass


def stop_worker(process):
    if process.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        if process.poll() is None:
            process.kill()
    process.wait()


def run_process(command, stage, env, report, resource_safety):
    report(f'{stage}…')
    with subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)) as process:
        lines = queue.Queue(maxsize=128)
        stop = threading.Event()

        def read(stream, structured):
            try:
                for line in stream:
                    if not enqueue((structured, line)):
                        return
            finally:
                enqueue(None)

        def enqueue(value):
            while not stop.is_set():
                try:
                    lines.put(value, timeout=.2)
                    return True
                except queue.Full:
                    pass
            return False

        readers = [threading.Thread(target=read, args=(stream, structured), daemon=True)
                   for stream, structured in ((process.stdout, True), (process.stderr, False))]
        for reader in readers:
            reader.start()
        events, failure = [], None
        try:
            import psutil
            try:
                guard = ProcessingGuard(process.pid, resource_safety=resource_safety)
            except psutil.NoSuchProcess:
                guard = None
            deadline = time.monotonic() + .5
            remaining = len(readers)
            while remaining:
                if guard is not None and time.monotonic() >= deadline:
                    message = guard.check()
                    if message:
                        raise WorkerError(message)
                    deadline = time.monotonic() + .5
                try:
                    item = lines.get(timeout=max(.01, deadline - time.monotonic()) if guard else .5)
                except queue.Empty:
                    continue
                if item is None:
                    remaining -= 1
                    continue
                structured, line = item
                if guard is not None:
                    guard.output()
                line = line.rstrip()
                if not structured or not line.startswith(EVENT_PREFIX):
                    if line:
                        report(line)
                    continue
                try:
                    event = json.loads(line[len(EVENT_PREFIX):])
                except ValueError:
                    report(line)
                    continue
                kind = event.get('type')
                if kind in RETAINED_EVENTS:
                    events.append(event)
                if kind in {'failed', 'blocked'}:
                    failure = event.get('message') or kind
                if event.get('message'):
                    report(str(event['message']))
            code = process.wait(timeout=10)
        except BaseException:
            stop_worker(process)
            raise
        finally:
            stop.set()
            for reader in readers:
                reader.join(timeout=1)
        if code or failure:
            detail = failure or f'worker exit code {code}'
            raise WorkerError(f'{stage} failed: {detail}')
        return events


@contextmanager
def worker_request(stage, args):
    if stage not in BATCH_STAGES:
        yield args
        return
    path = write_request(stage, args)
    try:
        yield ['--request', str(path), '--project', argument(args, '--project')]
    finally:
        remove_request(path)


def run_worker(stage, args, env, report):
    with worker_request(stage, args) as arguments:
        command = worker_command(stage, *arguments)
        return run_process(command, stage, env, report, env['RGB_MOCAP_RESOURCE_SAFETY'] == '1')

"""Bounded HTTPS recovery for exact pinned support archives; standard library only."""
from pathlib import Path
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request


class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Support download redirected outside HTTPS')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def valid(path, spec):
    return path.is_file() and path.stat().st_size == spec['bytes'] and digest(path) == spec['sha256']


def transient(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 416, 429, 500, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        return transient(exc.reason)
    if isinstance(exc, ssl.SSLCertVerificationError):
        return False
    # Local disk/permission errors and malformed responses must not be retried.
    return isinstance(exc, (ssl.SSLEOFError, ssl.SSLZeroReturnError, ConnectionError,
                            socket.timeout, socket.gaierror, http.client.IncompleteRead, http.client.RemoteDisconnected))


def download(cache, name, spec, *, open_url=None, clock=time.monotonic, sleep=time.sleep, cancel=None):
    if Path(name).name != name or name in ('', '.', '..') or ':' in name or '\\' in name:
        raise ValueError('Invalid support download name')
    url = spec['url']
    if urllib.parse.urlsplit(url).scheme != 'https':
        raise ValueError('Support download must use HTTPS')
    size, expected = spec['bytes'], spec['sha256']
    if not isinstance(size, int) or size <= 0 or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('Invalid support download pin')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (name + '.zip')
    partial = path.with_name(path.name + '.partial')
    metadata = partial.with_name(partial.name + '.json')
    pin = dict(url=url, bytes=size, sha256=expected)
    if valid(path, spec):
        return path

    def reset():
        partial.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)

    try:
        if json.loads(metadata.read_text('utf8')) != pin:
            reset()
    except (OSError, ValueError, TypeError):
        reset()
    metadata.write_text(json.dumps(pin), encoding='utf8')
    open_url = open_url or urllib.request.build_opener(HTTPSOnly()).open
    began = clock()
    # Large Blender archives get a size-aware total budget, independent of the
    # finite socket timeout. All retries share this budget.
    budget = max(180., min(7200., 120. + size / (128 * 1024)))

    def check():
        if cancel:
            cancel()
        if clock() - began >= budget:
            raise TimeoutError('Support download total time limit reached: ' + name)

    for attempt in range(3):
        check()
        start = partial.stat().st_size if partial.exists() else 0
        if start >= size:
            if valid(partial, spec):
                os.replace(partial, path)
                metadata.unlink(missing_ok=True)
                return path
            reset()
            metadata.write_text(json.dumps(pin), encoding='utf8')
            start = 0
        headers = {'User-Agent': 'RGBMoCap-Setup/1', 'Accept-Encoding': 'identity'}
        if start:
            headers['Range'] = 'bytes=%d-' % start
        print('Downloading %s from upstream (attempt %d/3, %.1f/%.1f MiB)...' %
              (name, attempt + 1, start / 1024**2, size / 1024**2), flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            check()
            with open_url(request, timeout=min(30., max(1., budget - (clock() - began)))) as response:
                if urllib.parse.urlsplit(response.geturl()).scheme != 'https':
                    raise ValueError('Support response is not HTTPS')
                status = response.status
                resume = status == 206
                if resume:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if (not match or int(match[1]) != start or int(match[3]) != size
                            or not start <= int(match[2]) < size or int(match[2]) != size - 1):
                        reset()
                        raise ValueError('Invalid support download Content-Range')
                elif status == 200:
                    start = 0
                else:
                    raise ValueError('Unexpected support download status: ' + str(status))
                count, last = start, clock()
                with partial.open('ab' if resume else 'wb') as output:
                    while True:
                        check()
                        # read1 returns available data rather than waiting to
                        # fill a large buffer during a slow, trickling transfer.
                        chunk = getattr(response, 'read1', response.read)(256 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > size:
                            raise ValueError('Support download exceeds pinned size')
                        output.write(chunk)
                        if clock() - last >= 2:
                            print('Downloading %s: %.1f/%.1f MiB (%.0f s)' %
                                  (name, count / 1024**2, size / 1024**2, clock() - began), flush=True)
                            last = clock()
                if count != size:
                    raise http.client.IncompleteRead(b'', size - count)
            check()
            if not valid(partial, spec):
                reset()
                raise ValueError(name + ' archive hash or size mismatch')
            os.replace(partial, path)
            metadata.unlink(missing_ok=True)
            return path
        except Exception as exc:
            if isinstance(exc, ValueError):
                reset()
            if not transient(exc) or attempt == 2:
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                raise
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                reset()
            metadata.write_text(json.dumps(pin), encoding='utf8')
            check()
            delay = 1. * (attempt + 1)
            if isinstance(exc, urllib.error.HTTPError):
                retry_after = exc.headers.get('Retry-After', '') if exc.headers else ''
                if retry_after.isdigit():
                    delay = min(10., max(delay, float(retry_after)))
                exc.close()
            print('Transient support download failure for %s (%s); retrying in %.0f s.' %
                  (name, type(exc).__name__, delay), flush=True)
            sleep(delay)

# Logging: ownership, size limits, rotation

Three problems, all now fixed:

1. log files were written by whichever uid happened to run the process, so files
   created by a container could not be appended to by a developer -- and
   `dictConfig` **raised on startup** when that happened;
2. there was **no size limit** of any kind -- plain `FileHandler`, mode `a`,
   growing until the disk filled (a `main.log` of 13.6 MB was sitting in the repo);
3. the Dockerfile `COPY`d the host's `./logs` into the image, baking those same
   foreign-owned files (14 MB) into every build.

## Avoiding the ownership clash

The clash is structural: the image creates `appuser` as **uid 1000**
(`Dockerfile: ARG uid=1000`), and a developer account is usually some other uid.
A file created by one is mode `644` owned by the other, so the second process
cannot append -- even in a world-writable directory. Four things address it:

**1. Never share the log path between host and container.** `docker-compose.yml`
mounts a *named volume* (`kggen-logs:/workspace/logs`), not a bind mount of
`./logs`. Docker owns the volume's contents; the host never writes there. Read
logs with `docker compose logs` or
`docker compose exec server cat /workspace/logs/main-api.log`.

> If you do bind-mount for local debugging, expect this problem and either match
> uids (`docker build --build-arg uid=$(id -u)`) or point `KGGEN_LOG_DIR`
> somewhere else.

**2. The log directory is configurable and absolute in the container.**
`KGGEN_LOG_DIR` (default `logs`) replaces a relative path that used to resolve
against whatever the working directory happened to be. Compose sets
`/workspace/logs`.

**3. Filenames are scoped by process role.** The API writes `main-api.log`, a
worker writes `main-worker.log`. Two processes sharing a volume never contend for
the same file -- which also matters because `RotatingFileHandler` is not
multi-process safe (see below).

**4. If it happens anyway, the process degrades instead of dying.** At startup
`app/logging_setup.py` probes both the directory *and* each target file. If either
is unusable it drops the file handlers, logs one warning, and continues on stdout:

```
WARNING - File logging disabled (cannot append to logs/main-api.log:
          [Errno 13] Permission denied); logging to stdout only
```

Losing file logs is an acceptable degradation; refusing to boot is not. Note the
per-file check is what makes this useful -- probing only the directory leaves a
handler that raises a `PermissionError` traceback on *every record* instead.

**5. Nothing host-generated enters the image.** `COPY ./logs` is replaced by
`RUN mkdir -p /workspace/logs`, and a new `.dockerignore` excludes `logs/`,
`*.log`, `.venv/` and other local artefacts from the build context.

## Size limits and rotation

Yes, rotation makes sense, and it is now on. File handlers are
`RotatingFileHandler`, size-based rather than time-based: the concern is bounded
disk, and log volume here is bursty per job rather than per day.

| Setting | Default | Meaning |
|---|---|---|
| `KGGEN_LOG_MAX_BYTES` | `10485760` (10 MB) | rotate a file once it exceeds this |
| `KGGEN_LOG_BACKUP_COUNT` | `3` | rotated copies kept (`.1`, `.2`, `.3`) |
| `KGGEN_LOG_DIR` | `logs` | where files go |
| `KGGEN_LOG_TO_FILES` | per-process (below) | write files at all, on top of stdout |

Worst case on disk is `MAX_BYTES × (1 + BACKUP_COUNT)` **per file** -- 40 MB
each, so ~120 MB for the three configured streams (main/access/uvicorn).

**Container stdout is bounded too.** Rotating files but leaving stdout unbounded
just moves the problem to the Docker host's `json-file` logs, so every compose
service now carries:

```yaml
logging:
  driver: json-file
  options: { max-size: "10m", max-file: "3" }
```

## Which processes write files

| Process | Files by default | Why |
|---|---|---|
| API (`app.server`) | **yes** (`main-api.log`, ...) | single process, so it safely owns its files |
| Worker (`app.worker`) | **no**, stdout only | replicas share one volume, and `RotatingFileHandler` is not multi-process safe: two workers rotating the same file clobber each other |

Override either with `KGGEN_LOG_TO_FILES=1` / `=0`. For workers that is only
advisable with a single replica; otherwise read them via
`docker compose logs worker`.

## Cleaning up the pre-existing files

The stale `logs/*.log` in a checkout are owned by uid 1000 and are not tracked by
git (`.gitignore` has `logs`). They are now inert -- nothing writes to those names
any more -- but to remove them:

```bash
sudo rm -f logs/main.log logs/access.log logs/uvicorn.log
# or, without sudo, from a container that owns them:
docker run --rm -v "$PWD/logs:/l" alpine sh -c 'rm -f /l/*.log'
```

## Implementation notes

`app/logging.yaml` stays the declarative source of truth for formatters, levels
and logger→handler wiring, and declares **bare filenames** (`main.log`, not
`logs/main.log`). `app/logging_setup.configure_logging()` supplies everything
environment-dependent: it resolves `KGGEN_LOG_DIR`, converts the file handlers to
`RotatingFileHandler` with the configured limits, suffixes filenames with the
role, and drops them if unusable. A test asserts the YAML never re-embeds a
directory.

`tests/test_logging_setup.py` covers rotation limits, that rotation actually
bounds bytes on disk, role scoping, both fallback paths (unwritable directory and
unappendable file), and that the fallback emits no per-record traceback.

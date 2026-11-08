# Khatt — Cairo metro what-if

Draw a hypothetical metro line anywhere in Greater Cairo, click **Construct line**, and within about two seconds see how local businesses are affected: spending shifting between shopping streets, footfall around stations, rent pressure, and the effect on microbuses and street vendors.

How it works: [CLAUDE.md](CLAUDE.md) (design) and [DECISIONS.md](DECISIONS.md) (assumptions and choices).

## Run it

The app is two servers, each in its own terminal window or tab. Run both commands from the repository folder. On a fresh clone, do the [first-time setup](#first-time-setup) first.

| Server | Command | Address |
|---|---|---|
| Backend (FastAPI) | `make api` | http://127.0.0.1:8000 |
| Website (Vite + React) | `make web` | http://localhost:5173 |

**Terminal 1: start the backend.**

```bash
make api
```

It's ready when it prints `Application startup complete.` Leave this terminal open.

**Terminal 2: start the website.**

```bash
make web
```

When it shows `Local: http://localhost:5173/`, open that address in your browser.

**To stop**, press Ctrl+C in each terminal.

### Good to know

- **Edits reload on their own.** Saving a Python file restarts the backend, and saving a file in `web/src` refreshes the page.
- **"Address already in use", or the website starting on port 5174 instead of 5173,** means an old copy is still running. The backend only accepts requests from port 5173, so free both ports and start again:

  ```bash
  lsof -ti tcp:8000,5173 -sTCP:LISTEN | xargs kill
  ```

- **The page loads but shows no data:** the backend isn't running or hasn't finished starting. Check terminal 1, then refresh the page.

## First-time setup

You need [uv](https://docs.astral.sh/uv/), Node.js 18 or newer, and [Git LFS](https://git-lfs.com). uv installs Python 3.11 for you.

1. Fetch the data stored in Git LFS:

   ```bash
   git lfs install && git lfs pull
   ```

2. Install the Python environment (`.venv`) and the website's packages:

   ```bash
   make setup
   ```

3. Rebuild the walking network (`data/processed/walk_*`), which is not in git. This runs the data pipeline and downloads its one input that is not in LFS, the full Egypt OSM extract (about 180 MB):

   ```bash
   make data
   ```

4. Build the travel-time matrix (`data/processed/T0.npy`), which is not in git either:

   ```bash
   make baseline
   ```

   It runs in the background and took about 25 minutes on an Apple Silicon Mac. Follow it with `tail -f logs/baseline.log` and start the backend once it finishes. It uses r5py, which needs a Java JDK; if Java isn't found, copy `.env.example` to `.env` and set `JAVA_HOME`. If r5py fails, the script falls back to a simpler scipy model; `make baseline-fallback` uses that model directly and takes minutes.

### Without uv (pip and requirements.txt)

Don't copy a `.venv` folder to another machine: it holds absolute paths and binaries built for the computer that made it. Recreate it instead from `requirements.txt`, which pins the same versions as `uv.lock`. Use Python 3.11 and run these from the repository folder:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

The `make` targets that run Python call uv, so start the backend with this in place of `make api`:

```bash
.venv/bin/uvicorn khatt.api.main:app --reload --port 8000
```

`make web` doesn't need uv. For other targets, `make -n <target>` prints the commands; run them with `.venv/bin/python` in place of `uv run --frozen python`. On Windows, use `.venv\Scripts\` in place of `.venv/bin/`.

After changing dependencies in `pyproject.toml`, run `uv lock` and regenerate `requirements.txt` with the command at the top of that file.

### Try it without the real data

Run `make api-synthetic` in place of `make api` to start the backend on an invented city, which builds in seconds. It's handy for working on the website before the real data is ready.

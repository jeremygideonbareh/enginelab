# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub's
[security advisories](https://github.com/jeremygideonbareh/enginelab/security/advisories/new). We aim to
reply within a week and to publish a fix and an advisory together.

## Scope

EngineLab parses untrusted files (STEP, IGES, SolidWorks, ZIP, meshes). Issues we especially want to hear
about:

- a file that crashes the server, uses unbounded memory or CPU, or hangs it;
- path traversal from archives (ZIP uploads are extracted with a guard; tell us if you get around it);
- anything that lets one user of a shared server see another user's uploads or results.

## Running a public server

The built-in server is designed for local use. If you host it for others:

- set `ENGINELAB_BETA=1` (upload size limit, job and optimiser limits);
- put it behind HTTPS (a reverse proxy or a tunnel), bound to `127.0.0.1`;
- do not set `HF_TOKEN` / `MOUSER_API_KEY` unless you need them, and never commit them.

Trained models are loaded with `joblib` (pickle). Only load model files from this project's releases or
that you trained yourself; `tools/download_models.py` checks their SHA-256 before use.

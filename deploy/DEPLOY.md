# Publishing the EngineLab beta

The beta runs as a Docker container. The recommended host is a **Hugging Face Space**: the free CPU tier
(2 vCPU, 16 GB RAM) is enough, and it builds straight from the files below. Any Docker host also works
(Render, Fly.io, a VPS), as long as it has at least 4 GB RAM; OpenCASCADE and the models need it.

## 1. Build the publishable folder

```powershell
python tools/build_space.py
```

This writes `deploy/space/` with only:

- the code;
- the current trained models (compressed);
- the three permissively licensed sample robots (MIT, Apache-2.0, BSD-3);
- the Dockerfile.

It never includes your own CAD, the GPL Voron files, the test engines or caches of private files.

## 2. Hugging Face Space

1. Create a free account at huggingface.co. Then make an access token at Settings → Access Tokens, with type "Write".
2. Log in once on this computer, in a terminal:

   ```powershell
   hf auth login
   ```

3. Create the Space and upload. Use `--private` to share it only with people you invite:

   ```powershell
   hf repo create enginelab-beta --repo-type space --space_sdk docker
   ```

   ```powershell
   hf upload <your-username>/enginelab-beta deploy/space . --repo-type space
   ```

4. The Space builds, which takes about 5–10 minutes the first time, and starts at
   `https://huggingface.co/spaces/<your-username>/enginelab-beta`.

To update it later, rebuild the folder and run the same `hf upload` command again.

## 3. Local check of the container (optional)

```powershell
docker build -t enginelab-beta deploy/space
```

```powershell
docker run --rm -p 7860:7860 enginelab-beta
```

Then open http://localhost:7860.

## Beta limits (set by `ENGINELAB_BETA=1` in the Dockerfile)

| Limit | Value |
|---|---|
| Upload size | 100 MB |
| Analyses running at once | 2 (others queue) |
| Assemblies kept in memory | 8 |
| Optimiser runs at once | 2 |
| Simulations per optimiser run | 3,000 |

Uploads are kept in memory only. Optimiser history is stored in the container and is lost when the Space restarts.

## Free option: public link from this laptop (Cloudflare quick tunnel)

Hugging Face now needs a PRO subscription for Docker Spaces. The free alternative runs the beta on
this laptop and gives it a public `https://<random-words>.trycloudflare.com` address:

```powershell
powershell -ExecutionPolicy Bypass -File deploy/start_public_beta.ps1
```

- It serves `deploy/space` (public samples only, beta limits on), bound to 127.0.0.1 so only the tunnel reaches it.
- It keeps the laptop awake while running; Ctrl+C (or closing the window) stops the server and the link.
- The address changes on every start, so send testers the new one.
- It needs `cloudflared.exe` (Cloudflare's signed release, 2026.10.7) in `%USERPROFILE%\tools\cloudflared\`.
- After code changes, run `python tools/build_space.py` first.

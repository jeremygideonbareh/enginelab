# Contributing to EngineLab

Thanks for helping. EngineLab is built for engineers, so the bar is simple: **results must be measured,
not claimed.** Every change that affects accuracy comes with a number before and after.

## Ways to help

- **Report a bug** with the issue template. Include the file type, its size, what you expected and what
  happened. If the problem is a specific CAD file, attach it only if you are allowed to share it.
- **Contribute labelled real CAD** under an open licence (MIT, Apache-2.0, BSD, CC-BY, CERN-OHL…).
  Engines are the most wanted: the engine classifier is limited by how little real engine CAD is
  openly available. Add the source and licence to `data/real_cad/SOURCES.md` in your pull request
  (the files themselves stay out of git; we download them from the source).
- **Improve a model or the physics.** Load models live in `enginelab/sim/parts.py`; the generator in
  `enginelab/generator.py` and `enginelab/std_generator.py`.
- **UI and docs.** The web app is plain HTML/CSS/JS in `web/` (Three.js + Chart.js from a CDN, no build step).

## Development setup

```bash
git clone https://github.com/jeremygideonbareh/enginelab.git
cd enginelab
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/download_models.py --samples
python -m pytest -q
python run.py                                         # http://127.0.0.1:8000
```

After editing `web/styles.css` or `web/app.js`, bump the `?v=` number on their links in
`web/index.html` so browsers fetch the new version.

## Pull requests

1. Open an issue first for anything larger than a bug fix, so we can agree on the approach.
2. Keep pull requests focused: one change, with tests.
3. `python -m pytest -q` must pass. Add a test for every bug you fix.
4. **Accuracy changes:** run the relevant evaluation and paste the before/after numbers in the PR:
   - engine classifier: `python tools/evaluate.py` and `python tools/eval_real_engines.py`
   - component classifier: `python -m enginelab.component_model ...` (leave-one-assembly-out report)
   - optimiser: `python tools/opt_bench.py`
5. Never commit CAD you are not allowed to redistribute, trained model files, caches, or anything
   personal. `.gitignore` already excludes `data/models/`, `data/real_cad/`, `data/cache/` and
   `data/feedback/`.

## Style

- Python 3.12+, standard library first, type hints where they help, docstrings that say *why*.
- Engineering formulas cite their source in a comment (e.g. "Shigley 6-8", "Heywood ch. 2").
- User-facing text is plain English, active voice, no jargon without an explanation.

## Licence of contributions

By contributing you agree that your contribution is licensed under the MIT licence of this project.

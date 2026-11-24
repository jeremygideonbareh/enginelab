FROM python:3.12-slim

# OpenCASCADE (cadquery-ocp) and gmsh need a few X/GL runtime libraries even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libxi6 libxkbcommon0 \
        libfontconfig1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    ENGINELAB_BETA=1 HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "run.py"]

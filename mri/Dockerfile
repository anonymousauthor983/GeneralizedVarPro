FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app/mri
COPY requirements.txt requirements-figures.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-figures.txt
COPY . .
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/matplotlib
CMD ["python", "reproduce.py", "--help"]

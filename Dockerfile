FROM verlai/verl:app-verl0.5-transformers4.55.4-sglang0.4.10.post2-mcore0.13.0-te2.2

RUN pip install --no-cache-dir polars
RUN pip install --no-cache-dir swanlab
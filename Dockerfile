FROM tensorflow/tfx:0.26.0
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
RUN pip install --no-cache-dir psutil

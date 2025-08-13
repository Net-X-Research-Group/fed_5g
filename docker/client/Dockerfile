FROM python:3.12-slim

# Set the working directory
WORKDIR /app
COPY ../../flwr-application /app/flwr-application
COPY ../../helper_scripts /app/helper_scripts
COPY ../../dist /app/dist
COPY ../../util /app/util
COPY docker/client/docker-entrypoint.sh /app/docker-entrypoint.sh

# Copy the requirements file into the container
COPY ../../requirements.txt /app/
RUN apt update && apt install -y \
    build-essential

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install /app/dist/flwr-1.20.0-py3-none-any.whl

ENV PYTHONPATH=/app

RUN mkdir -p /app/node_datasets
RUN mkdir -p /app/flwr_output

RUN chmod +x /app/docker-entrypoint.sh


ENTRYPOINT ["/app/docker-entrypoint.sh"]
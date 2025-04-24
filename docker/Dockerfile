From python:3.10-slim

# Set the working directory
WORKDIR /app
COPY ../flwr-application /app/flwr-application
COPY ../helper_scripts /app/helper_scripts
COPY ../dist /app/dist


# Copy the requirements file into the container
COPY ../requirements.txt /app/
RUN apt update && apt install -y \
    build-essential

RUN pip install /app/dist/flwr-1.15.2-py3-none-any.whl
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV PYTHONPATH=/app

RUN mkdir -p /app/node_datasets

COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh


ENTRYPOINT ["/app/docker-entrypoint.sh"]


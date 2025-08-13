FROM python:3.12-slim

# Set the working directory
WORKDIR /app
COPY ../../flwr-application /app/flwr-application
COPY ../../helper_scripts /app/helper_scripts
COPY ../../dist /app/dist
COPY ../../util /app/util

# Copy the requirements file into the container
COPY ../../requirements.txt /app/
RUN apt update && apt install -y \
    build-essential \
    curl

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install /app/dist/flwr-1.20.0-py3-none-any.whl

ENV PYTHONPATH=/app

RUN mkdir -p /app/flwr_output

# Make helper scripts executable
RUN chmod +x /app/helper_scripts/*.sh

ENTRYPOINT ["flower-superlink"]
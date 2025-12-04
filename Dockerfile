# Base: clean Ubuntu environment
FROM ubuntu:22.04

# Non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    supervisor \
    git \
    nano \
    build-essential \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------
# Install Ollama (official installation)
# ----------------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose Ollama port
EXPOSE 11434

# Install Python dependencies you need
RUN pip3 install supervisor
RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install gunicorn
RUN pip3 install python-multipart
RUN pip3 install pydantic
RUN pip3 install django==4.2
RUN pip3 install django-unfold
RUN pip3 install django-storages[azure]
RUN pip3 install psycopg2-binary

# Vision Search
RUN pip3 install faiss-cpu
RUN pip3 install clip-anytorch
RUN pip3 install Pillow

# Gemini
RUN pip3 install -q -U google-genai

# Tansformer cli
RUN pip3 install -U "huggingface_hub[cli]"

# Set Python path (optional)
ENV PYTHONPATH=/app


# ----------------------------------------
# Copy Supervisor config + entrypoint
# ----------------------------------------
COPY ./supervisord.conf /etc/supervisord.conf
COPY ./entrypoint.sh /home/entrypoint.sh

# Ensure script is executable and owned by root
RUN chmod +x /home/entrypoint.sh

# ----------------------------------------
# App directory
# ----------------------------------------
WORKDIR /app
COPY ./app /app

# ----------------------------------------
# Ports
# ----------------------------------------
EXPOSE 11434
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ----------------------------------------
# Entrypoint
# ----------------------------------------
ENTRYPOINT ["/bin/bash", "-c", "/home/entrypoint.sh"]

# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy your requirements file and install dependencies
COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync

# Copy your application code
COPY ./src ./src
COPY main.py .

# Expose the port uvicorn will run on (default 8000)
EXPOSE 8000

# Run the python file with uvicorn (adjust app:app to your app module and variable)
RUN ["source", ".venv/bin/activate"]
CMD ["uv", "run", "main.py"]
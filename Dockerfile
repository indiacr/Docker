# Use the official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libasound2 \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pygame gtts

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]

RUN apt-get update && apt-get install -y pulseaudio

ENV PULSE_SERVER=unix:/tmp/pulseaudio.socket
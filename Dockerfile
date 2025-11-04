# Use an official Python runtime as a parent image
FROM python:3.13-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Expose the port your application listens on (if it's a web app)
EXPOSE 8090

# Define the command to run your application
CMD ["python", "app.py"]
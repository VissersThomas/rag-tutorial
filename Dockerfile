FROM python:3.12

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and .env file
COPY . .

# Expose the port the app runs on
EXPOSE 8999

# Command to run the application
# Using the .env file that will be mounted or copied into the container
CMD ["python", "app.py"]

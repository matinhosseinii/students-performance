# 1. Start with an official Python base image
FROM python:3.13-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy files needed to install the project dependencies
# This leverages Docker's layer caching. This layer will only be rebuilt
# if these specific files change.
COPY pyproject.toml .
COPY requirements.txt . 
COPY src/ ./src

# 4. Install the project and its dependencies
# 'pip install .' reads pyproject.toml to install your project as a package.
# We also install from requirements.txt for any specific pinned versions.
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

# 5. Copy the rest of your application's files
# This includes the Streamlit app, models, notebooks, etc.
COPY . .

# 6. Expose the port that Streamlit runs on
EXPOSE 8501

# 7. Define the command to run your app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


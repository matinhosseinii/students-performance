# Install all Python dependencies from the locked requirements file

build-docker-image:
	docker build -t student-performance-app .

docker-run:
	docker run -p 8501:8501 student-performance-app

# Use make install in git bash to run it
# You can add some command shortcuts
# label-studio-ml-backend-roboflow

This guide describes the simplest way to start using ML backend with Label Studio.

This project provides a custom Label Studio ML backend that connects to the Roboflow API for object detection inference. It was bootstrapped using the official [Label Studio ML Backend template](https://github.com/HumanSignal/label-studio-ml-backend).

## Prerequisites

Before running, you need a Roboflow API key.

## Running with Docker (Recommended)

1.  **Create a `.env` file:** In the `roboflow_backend` directory (the same directory as `docker-compose.yml`), create a file named `.env` and add your Roboflow API key:
    ```dotenv
    # .env
    ROBOFLOW_API_KEY=your_actual_roboflow_api_key_here
    # Optional: Set to true to install test dependencies during build
    # TEST_ENV=false
    ```
    *Replace `your_actual_roboflow_api_key_here` with your key.*

2.  **Start Machine Learning backend:** Run the following command on `http://localhost:9090`:

    ```bash
    docker-compose up --build -d
    ```
    *(The `--build` flag is only strictly necessary the first time or after code changes, `-d` runs it in detached mode).*

3.  **Validate that backend is running:**

    ```bash
    $ curl http://localhost:9090/
    {"status":"UP"}
    ```

4.  **Connect to the backend from Label Studio:** Go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as the URL.


## Building from source (Advanced)

To build the ML backend from source, you first need to ensure the `.env` file exists as described in the "Running with Docker" section. Then, build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker:

1.  **Clone the repository.**
2.  **Set environment variables:** Export your Roboflow API key:
    ```bash
    export ROBOFLOW_API_KEY="your_actual_roboflow_api_key_here"
    ```
3.  **Install dependencies:**
    ```bash
    python -m venv ml-backend
    source ml-backend/bin/activate
    pip install -r requirements.txt
    # Optionally, install test requirements
    # pip install -r requirements-test.txt
    ```

4.  **Start the ML backend:**
    ```bash
    label-studio-ml start ./roboflow_backend
    ```

# Configuration
Parameters can be set via environment variables, typically defined in the `.env` file for Docker Compose usage.


The following common parameters are available:
- `ROBOFLOW_API_KEY` (**Required**): Your API key for Roboflow.
- `TEST_ENV`: Set to `true` during `docker-compose build` to install packages from `requirements-test.txt`.
- `BASIC_AUTH_USER`: Specify the basic auth user for the model server.
- `BASIC_AUTH_PASS`: Specify the basic auth password for the model server.
- `LOG_LEVEL`: Set the log level for the model server (e.g., `DEBUG`, `INFO`).
- `WORKERS`: Specify the number of Gunicorn workers for the model server.
- `THREADS`: Specify the number of Gunicorn threads per worker.
- `LABEL_STUDIO_URL`: URL of the Label Studio instance (required for accessing storage).
- `LABEL_STUDIO_API_KEY`: API key for the Label Studio instance (required for accessing storage).

# Customization

The ML backend can be customized by adding your own models and logic inside the `./roboflow_backend` directory. 
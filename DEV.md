# Developer Guide

This document outlines the development, versioning, and deployment process for the `kg-gen` application.

## 1. Local Development Setup

For local development, we use a `docker-compose.override.yml` file to enable features like live-reloading.

1.  **Check the override file**: Ensure `docker-compose.override.yml` in the root directory is configured for your local setup.

2.  **Start the services**: Launch the application using `docker-compose up`. Docker Compose automatically includes the override file.

    ```bash
    docker-compose up --build
    ```

    The server will now be running at `http://localhost:8000`, and any changes you make to the source code will trigger an automatic reload.

## 2. Releasing a New Version

The release process is currently a manual procedure.

> **Note**: The goal is to automate this with a GitLab CI/CD pipeline, but this is pending infrastructure setup. The manual steps below are a temporary measure.

Follow these steps to release a new version:

1.  **Update the Version Number**:
    Use the `uv version` command to bump the version in `pyproject.toml`.

    ```bash
    # Choose one of the following commands:
    uv version --bump patch   # (e.g., 1.0.0 -> 1.0.1)
    uv version --bump minor   # (e.g., 1.0.0 -> 1.1.0)
    uv version --bump major   # (e.g., 1.0.0 -> 2.0.0)
    ```

2.  **Run the Publish Script**:
    Execute the `publish.sh` script. This script will:
    - Read the new version from `pyproject.toml`.
    - Check if the image version already exists in the Nexus repository.
    - Update the image tag in `docker-compose.yml`.
    - Commit the version change.
    - Build the Docker image with the new version tag.
    - Push the image to the Nexus repository.
    - Create and push a new Git tag for the release.

    ```bash
    ./publish.sh
    ```

## 3. Deploying to a Remote Server

Deployment requires SSH access to the target machine.

1.  **SSH into the remote server**:
    ```bash
    ssh user@your-remote-server.com
    ```

2.  **Navigate to the application directory**:
    This directory should contain your `docker-compose.yml` and `.env` files.

3.  **Update the Docker Image**:
    Pull the new version of the image from the Nexus repository.

    ```bash
    docker-compose pull
    ```

4.  **Restart the Application**:
    Restart the services using the new image.

    ```bash
    docker-compose up -d
    ```

    The application is now updated and running with the new version.

#!/bin/sh
# Exit if any command fails
set -e

# Execute the container's main command (CMD)
exec uvicorn app.server:app --host 0.0.0.0 "$@"

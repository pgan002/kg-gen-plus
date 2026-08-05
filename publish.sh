set -ex

export PACKAGE_VERSION=$(uv version --short)
export $(grep -v '^#' .env | xargs)


exists=$(docker pull $IMAGE_NAME:$PACKAGE_VERSION > /dev/null && echo "y" || echo "n")
if [ "$exists" = "y" ]
then
    echo "The version $PACKAGE_VERSION already exists."
    exit 1
fi

# --- checks -----------------------------------------------------------------
# Everything below this block mutates something you cannot easily take back: a
# commit, an image pushed to the registry, and a tag. There is no CI pipeline
# behind this any more -- this script *is* the release -- so these checks are
# the only thing standing between a broken tree and a published image. Run them
# first, while failing still costs nothing. `set -e` aborts on the first failure.
#
# SKIP_TESTS=1 ./publish.sh escapes this, for the case where you have just run
# the suite yourself and only want the release mechanics.
if [ "${SKIP_TESTS:-0}" != "1" ]
then
    uv run ruff check app/ src/ tests/ mcp/
    uv run ruff format --check app/ src/ tests/ mcp/

    # The durable-job tests need a Redis. Without KGGEN_TEST_REDIS_URL they skip
    # silently, which would mean publishing the queue/worker code with its own
    # tests never run -- so start a throwaway one for the duration. Pick a port
    # unlikely to collide with a local Redis, and always clean it up.
    if [ "${SKIP_REDIS_TESTS:-0}" != "1" ]
    then
        REDIS_CID=$(docker run -d --rm -p 63799:6379 redis:7-alpine \
            redis-server --save "" --appendonly no)
        trap 'docker stop "$REDIS_CID" > /dev/null 2>&1 || true' EXIT
        # Give it a moment to accept connections before the suite starts.
        for _ in $(seq 1 20); do
            docker exec "$REDIS_CID" redis-cli ping > /dev/null 2>&1 && break
            sleep 0.5
        done
        KGGEN_TEST_REDIS_URL=redis://localhost:63799/0 uv run pytest -q
        docker stop "$REDIS_CID" > /dev/null
        trap - EXIT
    else
        uv run pytest -q
    fi
fi

# `docker compose build` only builds because docker-compose.override.yml supplies
# the build: section -- and that file is untracked, so it exists only on machines
# that happen to have one. Without it compose builds nothing and the push below
# would ship a stale (or missing) image, silently. Fail loudly instead.
docker compose config | grep -q "dockerfile: Dockerfile" || {
    echo "ERROR: no service defines a build (docker-compose.override.yml missing?)."
    echo "       'docker compose build' would be a no-op and the push would ship a stale image."
    echo "       Build directly instead:  docker build -t \$IMAGE_NAME:\$PACKAGE_VERSION ."
    exit 1
}

# --- release ----------------------------------------------------------------
git add pyproject.toml
# Updates every `image: ${IMAGE_NAME}:<version>` line, which is both the server
# and the worker -- they intentionally share one image and must not drift apart.
sed -i "s/\(image: \${IMAGE_NAME}:\)[0-9].*/\1${PACKAGE_VERSION}/" docker-compose.yml
git add docker-compose.yml
git commit -m 'Update version in docker-compose.yml'
docker compose build
docker push $IMAGE_NAME:$PACKAGE_VERSION
#docker push $IMAGE_NAME:latest

git tag -a "$PACKAGE_VERSION" -m "New version $PACKAGE_VERSION"
git push origin
git push origin --tags

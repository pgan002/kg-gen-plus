set -ex

export PACKAGE_VERSION=$(uv version --short)
export $(grep -v '^#' .env | xargs)


exists=$(docker pull $IMAGE_NAME:$PACKAGE_VERSION > /dev/null && echo "y" || echo "n")
if [ "$exists" = "y" ]
then
    echo "The version $PACKAGE_VERSION already exists."
    exit 1
fi

git add pyproject.toml
sed -i "s/\(image: \${IMAGE_NAME}:\)[0-9].*/\1${PACKAGE_VERSION}/" docker-compose.yml
git add docker-compose.yml
git commit -m 'Update version in docker-compose.yml'
docker compose build
docker push $IMAGE_NAME:$PACKAGE_VERSION
#docker push $IMAGE_NAME:latest

git tag -a "$PACKAGE_VERSION" -m "New version $PACKAGE_VERSION"
git push origin
git push origin --tags

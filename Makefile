fullname=bmxai/tensorrt:latest

UID=$(shell id -u)
GID=$(shell id -g)
USER_NAME=$(shell id -un)

.PHONY: build
build: tensorrt

tensorrt: models/tensorrt/Dockerfile
	DOCKER_BUILDKIT=1 docker build \
		--build-arg UID=${UID} \
		--build-arg GID=${GID} \
		--build-arg USER_NAME=${USER_NAME} \
			-t ${fullname} -f models/tensorrt/Dockerfile .
run:
	docker run --privileged --gpus '"all"' --shm-size 10g \
			--rm -it --name bmx --ipc=host \
			--ulimit memlock=-1 \
			--ulimit stack=67108864 \
			--mount type=bind,src="${PWD}",target=/content/phi-playground \
			--workdir /content/phi-playground \
			${fullname} bash --login
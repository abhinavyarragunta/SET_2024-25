# How to use Docker


## Installation

Make sure you have Docker installed and make an account: https://docs.docker.com/engine/install/ 

Execute  make sure you have the application running and authenticated on your machine and that the Docker engine is running.
```bash
sudo docker login
``` 


## Making a Docker Image from a Local Machine
Make sure that you are working in a repository with a DockerFile, then run the following command to build a Docker Image.


```bash
docker build -t <local-image-name> .
docker tag <local-image-name>:<local-tag> mannvika/set:tag
```

## Making a Docker image from a DockerHub Repository
Pull the docker repository using this command:

```bash
docker pull mannvika/set:tag
```


## Creating/Running a Docker Container
If you need to, find the docker image name that you created by running:
```bash
docker images
```
Then run the docker, which creates a container using the following command:

```bash
sudo docker run --netowrk host --device /dev/video:/devideo0 -p 800:8000 mannvika/set:test
```

## Commiting Changes
After you have confirmed your changes, commit to your branch on github, navigate to the Actions tab, select the Build and Push Docker Image tab on the right-hand side, then click the Run workflow dropdown menu, select a branch to build, then click run workflow.

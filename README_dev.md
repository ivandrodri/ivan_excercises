Docker:

In case you experience some issues with the rendering on your host machine using docker
make sure to add the docker user to xhost. So run on your local machine: 

xhost +SI:localuser:root (in this case we run docker as root but replace root by your docker user)

To run the container just execute from the terminal:

docker-compose run my_app

and this will open up a bash session.

# ToDo: I have push the image to dockerhub so check if Dockerfile contains all the libraries

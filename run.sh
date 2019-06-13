docker build --tag=friendlyhello .

nvidia-docker run -it --mount type=bind,src=/home/levimcclenny/Repositories/MaterialsInformatics,dst=/app --name docker1 --rm friendlyhello

docker exec -it docker1 bash 
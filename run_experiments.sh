#!/bin/bash

gym_id=(

#MuJoCo
	"halfcheetah-random-v2"
	"hopper-random-v2"
	"walker2d-random-v2"
	"halfcheetah-medium-v2"
	"hopper-medium-v2"
	"walker2d-medium-v2"
	"halfcheetah-expert-v2"
	"hopper-expert-v2"
	"walker2d-expert-v2"
	"halfcheetah-medium-expert-v2"
	"hopper-medium-expert-v2"
	"walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"

#Maze2d

        "Maze2d-umaze-v1"
        "Maze2d-umaze-dense-v1"
        "Maze2d-medium-v1"
        "Maze2d-medium-dense-v1"

#Adroit
        "Pen-human-v1"
        "Door-human-v1"
        "Relocate-human-v1"
        "Hammer-human-v1"
        "Pen-cloned-v1"
        "Door-cloned-v1"
        "Relocate-cloned-v1"
        "Hammer-cloned-v1"


	)



for ((i=0;i<5;i+=1))
do 
	for gym_id in ${gym_id[*]}
	do
		python SQOG.py \
		--gym_id $gym_id \
		--seed $i \
		--beta 0.5 \
		--alpha 150

		python SQOG.py \
		--gym_id $gym_id \
		--seed $i \
		--beta 2.5 \
		--alpha 25
		
	done
done
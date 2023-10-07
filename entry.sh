#!/bin/bash

# Set the target directory
target_directory="/tianshou/offline_data"

# Create the target directory if it doesn't exist
#if [ ! -d "$target_directory" ]; then
#    mkdir "$target_directory"
#fi

mkdir -p "$target_directory/.d4rl"
mkdir -p "$target_directory/.minari"

# Create .d4rl directory if it doesn't exist
#if [ ! -d "$target_directory/.d4rl" ]; then
#    mkdir "$target_directory/.d4rl"
#fi

# Create .minari directory if it doesn't exist
#if [ ! -d "$target_directory/.minari" ]; then
#    mkdir "$target_directory/.minari"
#fi

# Check if symbolic links exist before creating them
if [ ! -L ~/.d4rl ]; then
    ln -s "$target_directory/.d4rl" ~/.d4rl
fi

if [ ! -L ~/.minari ]; then
    ln -s "$target_directory/.minari" ~/.minari
fi

exec "$@"
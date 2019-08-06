# Run The Commands Line by Line. Not All at Once (Requirement : Linux)

# Step0 : Install Gym-Retro & NEAT-python
## Step0-1 : Install Gym-Retro
git clone --recursive https://github.com/openai/retro.git gym-retro
cd gym-retro
pip install -e .

## Step0-2 : Install NEAT-Python
pip install neat-python
git clone https://github.com/drallensmith/neat-python.git
pip install opencv-python

## Step0-3 : Import The Game (SonicTheHedgehog-Genesis.md)
python gym-retro/scripts/import.py SonicTheHedgehog-Genesis.md

# Step1 : Start Training Sonic
## After the training is done,
## 'winner.pkl' file which contains the best performing genome will be created
python train_sonic_complexreward.py

# Step2 : play the result of the training
## The 'winner.pkl' is required to be in the same directory.
python playback.py

## Further technical details about the code are in the 'train_sonic_complexreward.py' script
## and the official NEAT-Python & Gym-Retro Documentation

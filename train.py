import os
from GAN import GAN


if os.path.isfile('models/generator0'):
    print("There are already existig models. Move them somewhere else.")
    exit()

TRAINING_STEPS = 2
TRAIN_STEP_SIZE = 5
BATCH_SIZE = 64

gans = []
for i in range(0, 10):
    gan = GAN()
    gan.initialize_models()
    gan.load_images(i)
    
    print("Training model %d" % i)
    for j in range(0, TRAINING_STEPS):
        gan.train(TRAIN_STEP_SIZE, BATCH_SIZE)
        gan.generator.save("models/generator%d-%05d" % (i, (j + 1) * TRAIN_STEP_SIZE))
    gan.generator.save("models/generator%d" % i)
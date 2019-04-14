import server
from GAN import GAN
import threading
print()
print()

print("Starting server")
server.start_server()

print("Creating GAN")
gan = GAN()
server.set_GAN(gan)

print("Initializing GAN")
gan.initialize_models()
gan.load_images()
#gan.train(iterations=1)


def train():
    gan.train(iterations=200, batch_size=64)

#x = threading.Thread(target=train)
#x.start()

input() #do not terminate
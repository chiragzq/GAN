import server
from GAN import GAN
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

input() #do not terminate
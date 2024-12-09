import asyncio
from plato.servers import fedavg

class Server(fedavg.Server):
    """Custom Plato server using specified IP."""
    def __init__(self):
        super().__init__()
        self.host = '129.105.6.252'
        self.port = 8000  # default Plato port

if __name__ == "__main__":
    server = Server()
    server.configure()
    
    # Run the asyncio event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.start())
    loop.close()

import ray
import time

@ray.remote
def f(data):
    # Function to store data in Ray's object store and return a reference
    return data

@ray.remote
class Owner:
    def __init__(self):
        self.ref = None

    def set_object_ref(self, data):
        # Store the object in Ray's object store using a remote task and return the reference
        # Two ways to create an object reference (in both cases, the owner is the caller)
        # 1. using ray.put
        # self.ref = ray.put(data)
        # 2 using a remote task
        self.ref = f.remote(data)
        return self.ref

    def warmup(self):
        # Dummy method to ensure the actor is alive
        return 0

@ray.remote
class Borrower:
    def get_object(self, ref):
        # Retrieve the object using the reference
        return ray.get(ref)

# Initialize Ray
ray.init()

# Create remote instances of Owner and Borrower
owner = Owner.remote()
borrower = Borrower.remote()

# Ensure the Owner actor is alive
ray.get(owner.warmup.remote())

# Set an object reference in the Owner using a remote task and get the reference
object_ref = owner.set_object_ref.remote(data="test1")

# Simulate some processing time
time.sleep(10)

# Retrieve the object using the Borrower and verify its content
data = ray.get(borrower.get_object.remote(object_ref))
assert data == "test1", "Data mismatch!"

# Terminate the Owner actor
ray.kill(owner)

# Wait for a short period to ensure the actor is terminated
time.sleep(2)

# Attempt to retrieve the object again, expecting failure
try:
    ray.get(borrower.get_object.remote(object_ref))
except Exception as e:
    print("Failed as expected after owner termination.")
    print(e)

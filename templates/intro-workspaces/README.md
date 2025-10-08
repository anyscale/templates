# Introduction to Workspaces

**⏱️ Time to complete**: 10 min

Welcome! You are currently in a Workspace, which is a persistent cloud IDE connected to a Ray cluster.

In this tutorial, you will learn:
1. Basic workspace features such as git repo persistence, cloud storage, and SSH authentication.
2. Ray cluster management features, such as adding multiple worker nodes.
3. Ray monitoring features such as viewing tasks in the dashboard.
4. Dependency management.

## "Hello world" in workspaces

Let's start by checking that Ray is working properly in your workspace. You can do this by running the following cell to execute a simple parallel Ray program.


```python
import ray

@ray.remote
def square(x):
    return x ** 2

futures = [square.remote(x) for x in range(100)]
results = ray.get(futures)
print("Success!", results)
```

## Workspace Basics

An Anyscale Workspace is a cloud IDE where you can develop and test Ray programs. Let's get started by creating a new git repo in the workspace's project directory (`/home/ray/default`). Workspaces will persist the tracked files in this git repo across restarts (as well as other files in the project directory).

We'll use the repo later on to author and run a simple Ray app.


```python
!mkdir my_repo && cd my_repo && git init
```

### Setting up SSH authentication (optional)

Anyscale generates a unique SSH key per user, which is accessible at `~/.ssh/id_rsa.pub`. If you'd like, you can [add this key to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) in order to access private repositories from the workspace.

The public key to add is outputted by the following command:


```python
!cat ~/.ssh/id_rsa.pub
```

### Cloud Storage

Workspace storage in the persisted project directory (`/home/ray/default`) is limited to 10 GB, so we recommend only using it to store git repos and smaller files. To learn more about options for files and storage, see [documentation](https://docs.anyscale.com/platform/workspaces/workspaces-files).

Cloud storage can be read and written from the workspace, as well as from any node in the Ray cluster.

Access built-in cloud storage using the `$ANYSCALE_ARTIFACT_STORAGE` URI as a prefix:


```python
import os
import re
import subprocess

class UnifiedStorageHandler:
    def __init__(self):
        self.artifact_storage = os.environ["ANYSCALE_ARTIFACT_STORAGE"]
        self.storage_type = self._detect_storage_type()
        self.setup_storage()
    
    def _detect_storage_type(self):
        """Detect storage type based on URI scheme"""
        if self.artifact_storage.startswith('s3://'):
            return 's3'
        elif self.artifact_storage.startswith('abfss://'):
            return 'abfss'
        elif self.artifact_storage.startswith('gs://'):
            return 'gcs'
        else:
            raise ValueError(f"Unsupported storage URI: {self.artifact_storage}")
    
    def setup_storage(self):
        """Setup storage-specific configurations"""
        if self.storage_type == 'abfss':
            # Convert ABFSS URI to blob storage URI for azcopy
            self.storage_uri = re.sub(
                r'^abfss://([^@]+)@([^.]+)\.dfs\.core\.windows\.net/', 
                r'https://\2.blob.core.windows.net/\1/', 
                self.artifact_storage
            )
            if self.storage_uri.endswith('/artifact_storage'):
                self.storage_uri = self.storage_uri[:-len('/artifact_storage')]
            
            # Login to Azure
            try:
                subprocess.run(['azcopy', 'login', '--identity'], check=True, capture_output=True)
                print("✓ Azure authentication successful")
            except subprocess.CalledProcessError as e:
                print(f"Azure login failed: {e}")
                
        elif self.storage_type == 's3':
            self.storage_uri = self.artifact_storage
            print("✓ Using S3 storage")
            
        elif self.storage_type == 'gcs':
            self.storage_uri = self.artifact_storage
            print("✓ Using Google Cloud Storage")
    
    def upload_file(self, local_path, remote_filename):
        """Upload file to storage"""
        remote_path = f"{self.storage_uri}/{remote_filename}"
        
        if self.storage_type == 's3':
            cmd = ['aws', 's3', 'cp', local_path, remote_path]
        elif self.storage_type == 'abfss':
            cmd = ['azcopy', 'copy', local_path, remote_path, '--from-to', 'LocalBlob']
        elif self.storage_type == 'gcs':
            cmd = ['gsutil', 'cp', local_path, remote_path]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Upload successful: {local_path} -> {remote_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Upload failed: {e}")
            print(f"Command: {' '.join(cmd)}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
            return False
    
    def download_file(self, remote_filename, local_path):
        """Download file from storage"""
        remote_path = f"{self.storage_uri}/{remote_filename}"
        
        if self.storage_type == 's3':
            cmd = ['aws', 's3', 'cp', remote_path, local_path]
        elif self.storage_type == 'abfss':
            cmd = ['azcopy', 'copy', remote_path, local_path]
        elif self.storage_type == 'gcs':
            cmd = ['gsutil', 'cp', remote_path, local_path]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✓ Download successful: {remote_path} -> {local_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            print(f"Command: {' '.join(cmd)}")
            if e.stdout:
                print(f"Stdout: {e.stdout}")
            if e.stderr:
                print(f"Stderr: {e.stderr}")
            return False

# Initialize the storage handler
storage = UnifiedStorageHandler()
print(f"Detected storage type: {storage.storage_type}")
print(f"Storage URI: {storage.storage_uri}")

# Create test file and upload
!echo "hello world" > /tmp/input.txt
storage.upload_file('/tmp/input.txt', 'saved.txt')
```


```python
# Download the file using the unified storage handler
if storage.download_file('saved.txt', '/tmp/output.txt'):
    # Read and display the downloaded file
    with open('/tmp/output.txt', 'r') as f:
        content = f.read()
        print(content)
else:
    print("Failed to download file")
```

## Ray cluster management

This workspace is connected to a Ray cluster. Click on the resources bar on the top right corner of the screen to open the cluster control panel. This panel shows a summary of Ray resource utilization, and you can use this panel to configure the cluster resources.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/resource-panel.png" height=400px />

### Configuring the Workspace node

The workspace node is the machine this notebook is running inside. You may wish to change the instance type of the workspace node specifically, e.g., to increase the available memory or add a GPU. Click the pencil icon in order to change the workspace node. Note that changing the workspace node will restart the workspace IDE.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/edit-workspace-node.png" height=300px />
<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/edit-workspace-dialog.png" width=400px />

### Adding worker nodes

To parallelize beyond the resources available to the workspace node, add additional worker nodes to the Ray cluster. Click "Add a node type" to add a number of nodes of a certain type to the cluster. While most use cases only require a single worker node type, you can add multiple distinct node types (e.g., high-CPU and GPU nodes) to the workspace as well.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/add-node-type.png" height=300px />
<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/add-node-dialog.png" height=300px />

### Using "Auto-select workers" mode

To let Ray automatically select what kind of worker nodes to add to the cluster, check the "Auto-select workers" box. Ray will add worker nodes as needed to run submitted tasks and actors. In auto mode, you cannot configure workers, but the resources panel will show which nodes have been launched.

We recommend using auto mode if you do not have specific cluster requirements, and are ok with waiting for the autoscaler to add nodes on-demand to the cluster.

## Monitoring Ray applications

In this section, we'll author a simple Ray python script and go over the tools available to monitor its execution. Let's take the opportunity to create a `my_app.py` file in the `my_repo` git repo you created earlier.

You can click on the "File Explorer" in the left pane of VSCode to create the new file.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/file-explorer.png" height=400px />

Double click this cell to copy & paste the following program into the file :

```python
import ray, time

@ray.remote
def do_some_work():
    print("Doing work")
    time.sleep(5)
    return "Done"

ray.get([do_some_work.remote() for _ in range(100)])
````

Then, use the next cell or the VSCode terminal to run the file:


```python
!python my_repo/my_app.py
```

### Understanding Ray log output

After running `my_app.py`, you should see output of the form `(do_some_work pid=29848) Doing work [repeated 4x across cluster]`. The prefix of the log message shows the function name, PID of the worker that ran the function, and if run on a remote worker, the node IP.

The result of the log message contains stdout and stderr from the function execution. Ray will also deduplicate repetitive logs from parallel execution of functions across the cluster.

### Monitoring program execution

Depending on the cluster size, the above script may take some time to run. Try playing around with the number of worker machines, increasing the sleep time, or the number of function calls. Use the tools overviewed below to understand how Ray parallelizes the program.

Let's overview some of the tools available to monitor Ray program execution in workspaces.

**Resources Panel**

The resources panel provides basic stats about cluster utilization, as well as an indication of which worker nodes are being used. Use the resource panel as a quick overview of cluster status before diving deeper into the Ray dashboard.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/resources-panel-stats.png" height=400px />

**Ray dashboard > Jobs**

To see the status of an active or previously run Ray job, navigate to `Ray Dashboard > Jobs` in the UI. Click a job to open the detail page where you can see an overview of job progress, logs, etc.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/ray-dashboard-jobs.png" height=400px />

**Ray dashboard > Metrics**

View the aggregate time-series metrics for the cluster in order to diagnose job execution efficiency. The `Ray Dashboard > Metrics` page offers metrics on Ray tasks, actors, as well as hardware resource utilization of the cluster.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/ray-dashboard-metrics.png" height=400px />

**Logs Tab**

View and search over Ray cluster and application logs in the Logs tab.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/logs-tab.png" height=400px />

## Dependency Management

In order to run code across a cluster, Ray ships code and other library dependencies to other machines in [runtime envs](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html). In workspaces, the code and installed PyPI packages are automatically added to the runtime env to be used by Ray.

To try this out, run the following command to install the `emoji` package. You'll see a notification that the package has been registered with the cluster.


```python
!pip install emoji
```

Navigate to the `Dependencies` tab of the workspace, and you should see the `emoji` package in the list there. You can use this UI to edit the workspace runtime dependencies, or the UI.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-workspaces/assets/dependencies-tab.png" height=400px />

Run the following cell to check that the `emoji` package is successfully installed on the cluster (to check this properly, make sure the cluster has at least one worker node added).


```python
import ray
import emoji
import time
import os

# Reset the Ray session in the notebook kernel to pick up new dependencies.
if ray.is_initialized():
    ray.shutdown()

@ray.remote
def f():
    my_emoji = os.environ.get("MY_EMOJI", ":thumbs_up:")
    print(emoji.emojize(f"Dependencies are {my_emoji}"))
    time.sleep(5)

ray.get([f.remote() for _ in range(100)])
print("Done")
```

### Cluster env vars

The dependencies tab also lets you set environment variables on cluster workers. Try it out by setting the ``MY_EMOJI`` env var and running the cell above again.

Note that this does not set the env var in VSCode, only on the cluster workers.


```python
# First set MY_EMOJI=:palm_tree: in the Workspace > Dependencies tab.
# The code below should then pick up your newly set env var.

# Note: need to reinitialize Ray to clear worker state for this notebook.
if ray.is_initialized():
   ray.shutdown()

ray.get([f.remote() for _ in range(100)])
```

That's it! Now you know everything you need to build scalable Ray applications in Anyscale Workspaces. Check out the template gallery and Ray documentation to learn more about what you can do with Ray and Anyscale.

## Summary

This notebook:
- Set up a basic development project in a workspace.
- Showed how to use different types of persistent storage.
- Demoed how to build and debug basic Ray application.



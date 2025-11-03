# Ray Log Files - Complete Guide

## Overview

Ray generates multiple log files to track system operations and application execution. This guide provides a comprehensive overview of all log files, organized by node type and purpose.

## Log Directory Structure

```
/tmp/ray/
├── session_latest/  (symlink to latest session)
│   └── logs/
│       ├── [application logs]
│       ├── [system logs]
│       └── events/
│           └── [event logs]
├── session_2023-05-14_21-19-58_128000_45083/
│   └── logs/
└── session_2023-05-15_21-54-19_361265_24281/
    └── logs/
```

> **Note:** Default location is `/tmp/ray/session_*/logs`. Customize with `ray.init()` or `ray start`.



## Ray Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           HEAD NODE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐            │
│  │ GCS Server   │  │  Dashboard   │  │   Autoscaler    │            │
│  │ (Metadata)   │  │   (Web UI)   │  │   (Scaling)     │            │
│  └──────────────┘  └──────────────┘  └─────────────────┘            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Raylet (Scheduler + Object Store)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Workers    │  │  IO Workers  │  │ Runtime Env  │               │
│  │(Tasks/Actors)│  │  (Spill/     │  │    Agent     │               │
│  │              │  │   Restore)   │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         WORKER NODE(S)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Dashboard    │                                                   │
│  │   Agent      │                                                   │
│  └──────────────┘                                                   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Raylet (Scheduler + Object Store)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Workers    │  │  IO Workers  │  │ Runtime Env  │               │
│  │(Tasks/Actors)│  │  (Spill/     │  │    Agent     │               │
│  │              │  │   Restore)   │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```



## Log Files by Node Type

### 🎯 HEAD NODE ONLY

These logs only exist on the head node:

| Log File | Purpose | File Types |
|----------|---------|------------|
| `gcs_server.[out\|err]` | Global Control Service - manages cluster metadata | stdout/stderr |
| `dashboard.[log\|out\|err]` | Dashboard web UI server | logger + stdout/stderr |
| `monitor.[log\|out\|err]` | Autoscaler managing cluster scaling | logger + stdout/stderr |
| `dashboard_[module_name].[log\|out\|err]` | Dashboard child processes (per module) | logger + stdout/stderr |

### 🔄 EVERY NODE (Head + Workers)

These logs exist on every node in the cluster:

| Log File | Purpose | File Types |
|----------|---------|------------|
| `raylet.[out\|err]` | Local scheduler and object store manager | stdout/stderr |
| `dashboard_agent.[log\|out\|err]` | Dashboard agent (one per node) | logger + stdout/stderr |
| `log_monitor.[log\|out\|err]` | Streams logs to driver | logger + stdout/stderr |
| `runtime_env_agent.[log\|out\|err]` | Manages runtime environments | logger + stdout/stderr |
| `worker-[worker_id]-[job_id]-[pid].[out\|err]` | Python/Java task and actor output | stdout/stderr |
| `java-worker*.log` | Java worker logs (if using Java) | logger |
| `python-core-driver-[worker_id]_[pid].log` | C++ core for Python/Java drivers | logger |
| `python-core-worker-[worker_id]_[pid].log` | C++ core for Python/Java workers | logger |
| `io-worker-[worker_id]-[pid].[out\|err]` | Object spill/restore workers | stdout/stderr |
| `runtime_env_setup-[job_id].log` | Runtime environment installation | logger |


## Application Logs

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Ray Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Job Submission (Ray Jobs API)                              │
│       ↓                                                     │
│  job-driver-[submission_id].log ────────────┐               │
│                                             │               │
│  Driver Process                             │               │
│       ↓                                     │               │
│  worker-[id]-[job_id]-[pid].out ────────────┤               │
│  worker-[id]-[job_id]-[pid].err ────────────┤               │
│                                             │               │
│  Task/Actor Processes                       │               │
│       ↓                                     │               │
│  worker-[id]-[job_id]-[pid].out ────────────┤               │
│  worker-[id]-[job_id]-[pid].err ────────────┤               │
│                                             │               │
│  Runtime Environment Setup                  │               │
│       ↓                                     │               │
│  runtime_env_setup-[job_id].log ────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Application Log Files

1. **`job-driver-[submission_id].log`**
   - Stdout of jobs submitted via Ray Jobs API
   - **Location:** Node where job was submitted

2. **`worker-[worker_id]-[job_id]-[pid].[out|err]`**
   - Python/Java drivers and workers
   - Captures all stdout/stderr from tasks and actors
   - `.out` = stdout + stderr
   - `.err` = stderr only
   - **Location:** All nodes running tasks/actors

3. **`runtime_env_setup-[job_id].log`**
   - Runtime environment installation logs (pip install, conda, etc.)
   - Only created when runtime environments are used
   - **Location:** Nodes where runtime environment is installed

4. **`runtime_env_setup-ray_client_server_[port].log`**
   - Runtime environment setup logs when using Ray Client
   - **Location:** Head node (Ray Client server)



## System Component Logs

### Core System Components

#### **`raylet.[out|err]`**
- Local scheduler managing task execution and object store
- **Location:** EVERY NODE
- **Rotation:** No

#### **`gcs_server.[out|err]`** 🎯
- Global Control Service managing cluster metadata
- **Location:** HEAD NODE ONLY
- **Rotation:** No

#### **`python-core-driver-[worker_id]_[pid].log`**
- C++ core logs for Ray drivers
- Ray drivers = Python/Java frontend + C++ core
- **Location:** Nodes running drivers
- **Rotation:** Yes

#### **`python-core-worker-[worker_id]_[pid].log`**
- C++ core logs for Ray workers
- **Location:** All nodes with workers
- **Rotation:** Yes


### Dashboard & Monitoring

```
Head Node                          Worker Nodes
┌─────────────┐                   ┌─────────────┐
│ Dashboard   │◄──────────────────│  Dashboard  │
│   Server    │      Reports      │    Agent    │
│             │◄──────────────────│             │
└─────────────┘                   └─────────────┘
      │                                  │
      ├─► dashboard.log                  ├─► dashboard_agent.log
      ├─► dashboard.out                  ├─► dashboard_agent.out
      └─► dashboard.err                  └─► dashboard_agent.err
```

#### **`dashboard.[log|out|err]`** 🎯
- Ray Dashboard web UI server
- `.log` = structured logger output
- `.out/.err` = stdout/stderr (usually empty unless crashes)
- **Location:** HEAD NODE ONLY

#### **`dashboard_agent.[log|out|err]`**
- Dashboard agent collecting metrics/logs from each node
- One agent per node
- **Location:** EVERY NODE

#### **`dashboard_[module_name].[log|out|err]`** 🎯
- Dashboard child process logs (one per module)
- Examples: `dashboard_job.log`, `dashboard_reporter.log`
- **Location:** HEAD NODE ONLY (typically)

#### **`log_monitor.[log|out|err]`**
- Streams logs from workers to driver
- **Location:** EVERY NODE

#### **`monitor.[log|out|err]`** 🎯
- Autoscaler logs for cluster scaling decisions
- **Location:** HEAD NODE ONLY


### Runtime Environment

```
┌─────────────────────────────────────────────────────────┐
│               Runtime Environment Lifecycle             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Request runtime env (pip, conda, containers)        │
│          ↓                                              │
│  2. runtime_env_agent.log ─────► Agent handles request  │
│          ↓                                              │
│  3. runtime_env_setup-[job_id].log ─► Installation      │
│          ↓                             logs (pip,       │
│  4. Environment ready                  conda, etc.)     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### **`runtime_env_agent.[log|out|err]`**
- Manages runtime environment creation, deletion, caching
- One agent per node
- **Location:** EVERY NODE

#### **`runtime_env_setup-[job_id].log`**
- Detailed installation logs (pip output, conda output, etc.)
- Only present when runtime environments are used
- **Location:** Nodes where environment is installed



### I/O Workers

#### **`io-worker-[worker_id]-[pid].[out|err]`**
- IO workers for spilling/restoring objects to external storage
- Created automatically from Ray 1.3+
- **Location:** Nodes performing object spilling


## Event Logs

Event logs are stored in the `events/` subdirectory and contain structured event data.

```
logs/
└── events/
    ├── event_GCS.log
    ├── event_RAYLET.log
    ├── event_CORE_WORKER_[pid].log
    ├── event_AUTOSCALER.log
    ├── event_EXPORT_DRIVER_JOB.log
    ├── event_EXPORT_ACTOR.log
    └── event_EXPORT_TASK_[pid].log
```

### Event Log Files

| Log File | Source | Location |
|----------|--------|----------|
| `event_GCS.log` | GCS server events | HEAD NODE |
| `event_RAYLET.log` | Raylet events | EVERY NODE |
| `event_CORE_WORKER_[pid].log` | Core worker events (per process) | EVERY NODE |
| `event_AUTOSCALER.log` | Autoscaler events | HEAD NODE |
| `event_EXPORT_DRIVER_JOB.log` | Export events for driver jobs | Varies |
| `event_EXPORT_ACTOR.log` | Export events for actors | Varies |
| `event_EXPORT_TASK_[pid].log` | Export events for tasks | Varies |


## Log Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Ray Application                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─► print(), logging ─────┐
             │                          │
             ├─► Task execution ────────┤
             │                          │
             └─► Actor methods ─────────┤
                                        ↓
                     ┌──────────────────────────────────────┐
                     │    Worker Process                     │
                     │    worker-[id]-[job].out              │
                     └──────────────┬───────────────────────┘
                                    │
                     ┌──────────────┼──────────────┐
                     ↓              ↓              ↓
              ┌────────────┐ ┌───────────┐ ┌────────────┐
              │   Local    │ │    Log    │ │   Driver   │
              │    File    │ │  Monitor  │ │  (stdout)  │
              └────────────┘ └───────────┘ └────────────┘
                                    │
                                    ↓
                          ┌──────────────────┐
                          │  External Logger │
                          │  (FluentBit,     │
                          │   Vector, etc.)  │
                          └──────────────────┘
```

## Quick Reference: Finding Logs by Use Case

Use this section to quickly locate the right logs and messages. Each item lists where to look and what to search for.

### My task is failing
- **Where to look**: `worker-[worker_id]-[job_id]-[pid].out` (stdout), `worker-[worker_id]-[job_id]-[pid].err` (errors)
- **Also check**: `python-core-worker-[worker_id]_[pid].log` (C++ core crashes)
- **Search for**: Tracebacks, import errors, segmentation faults, connection errors

### My job won't start
- **Where to look**: `job-driver-[submission_id].log`
- **Also check**: `runtime_env_setup-[job_id].log` (env install), `raylet.out` (scheduling)
- **Search for**: Dependency install failures, “Failed to put object…”, scheduling/backpressure warnings

### Cluster isn’t scaling
- **Where to look**: `monitor.[log|out|err]` (head node)
- **Also check**: `events/event_AUTOSCALER.log` (head node)
- **Search for**: “Scaling up/down…”, node launch failures, cloud quota limits

### Dashboard isn’t working
- **Where to look**: `dashboard.[log|out|err]` (head node)
- **Also check**: `dashboard_agent.[log|out|err]` (every node), `dashboard_[module].log`
- **Search for**: Port binding errors, HTTP 5xx, module crashes

### A node crashed
- **Where to look**: `raylet.[out|err]` (on that node), `gcs_server.[out|err]` (head node)
- **Also check**: `events/event_RAYLET.log`, `events/event_GCS.log`
- **Search for**: Process crashes, heartbeat timeouts, disconnections

### Object store is full or spilling to disk
- **Where to look**: `raylet.out` (spilling), `io-worker-[worker_id]-[pid].out` (I/O workers)
- **Also check**: `worker-*.err` or `python-core-worker-*.log` (ObjectStoreFullError)
- **Search for**:
  - “:info_message:Spilled … MiB…” (INFO) — normal spill progress
  - “Shared memory store full, falling back to allocating from filesystem”
  - “Out-of-disk: Failed to create object …” (critical)

### Workers are being killed due to memory (OOM)
- **Where to look**: `raylet.out` (Ray memory monitor), system logs via `dmesg`/`journalctl` (Linux OOM killer)
- **Also check**: `worker-*.err` for “UNEXPECTED_SYSTEM_EXIT” or “OutOfMemoryError” on `ray.get`
- **Search for**:
  - Ray memory monitor: “Killing worker with task … Memory on the node … exceeds the threshold”
  - Periodic summary: “Workers … killed due to memory pressure (OOM)”
  - Linux OOM: “killed process” in `dmesg`
  - See detailed OOM guidance below for remediation and configuration


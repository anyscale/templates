version 1.0

# Verbatim from broadinstitute/long-read-pipelines (wdl/structs/Structs.wdl),
# licensed BSD-3-Clause; see wdl/LICENSE.
#
# The struct is byte-identical to upstream's; only this comment is added.
#
# Every task in wdl/tasks/ takes an optional RuntimeAttr so callers can re-size it
# without editing the task, which is how the pipeline's inputs.*.json bring the
# vendored defaults down to something a small Ray cluster can schedule. Those
# defaults are per task and are sized for the Broad's fleet: Flye asks for 16 cores
# and 100 GiB, Quast 16 and 80, Medaka 8 and 24, AlignAsPAF 4 and 40. On Cromwell
# each of those provisions a VM to order; on Ray they are requests against nodes
# that already exist, and a request no node can satisfy waits rather than failing.

struct RuntimeAttr {
    Float? mem_gb
    Int? cpu_cores
    Int? disk_gb
    Int? boot_disk_gb
    Int? preemptible_tries
    Int? max_retries
    String? docker
}

struct DataTypeParameters {
    Int num_shards
    String map_preset
}

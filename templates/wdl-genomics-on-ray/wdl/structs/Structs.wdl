version 1.0

# Verbatim from broadinstitute/long-read-pipelines (wdl/structs/Structs.wdl),
# licensed BSD-3-Clause; see wdl/LICENSE.

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

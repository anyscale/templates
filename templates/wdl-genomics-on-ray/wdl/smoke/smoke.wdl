version 1.0

# Exercises the WDL <-> Ray contract: scatter, resource requests, File I/O and glob(), a
# gathering task, and stderr tailing. A stock debian image, so it runs in seconds anywhere.

workflow Smoke {
    meta {
        description: "Minimal end-to-end check of the Ray container backend."
    }
    parameter_meta {
        shards:    "how many parallel tasks to fan out to"
        greeting:  "text each shard echoes, to prove inputs reach the container"
        docker:    "image for every task; must exist for your host architecture"
    }

    input {
        Int shards = 4
        String greeting = "hello from WDL on Ray"
        String docker = "docker.io/library/debian:12-slim"
    }

    call MakeSeeds { input: shards = shards, docker = docker }

    scatter (seed in MakeSeeds.seeds) {
        call Shard {
            input:
                seed = seed,
                greeting = greeting,
                docker = docker
        }
    }

    call Gather {
        input:
            shard_reports = Shard.report,
            checksums = Shard.checksum,
            docker = docker
    }

    output {
        Array[File] shard_reports = Shard.report
        Array[Array[File]] shard_chunks = Shard.chunks
        File summary = Gather.summary
        Int total_lines = Gather.total_lines
    }
}

task MakeSeeds {
    input {
        Int shards
        String docker
    }

    command <<<
        set -euo pipefail
        seq 1 ~{shards}
    >>>

    output {
        Array[String] seeds = read_lines(stdout())
    }

    runtime {
        cpu: 1
        memory: "512 MiB"
        docker: docker
    }
}

task Shard {
    input {
        String seed
        String greeting
        String docker
    }

    # Not the default 1, so a scatter visibly consumes 2N CPUs in Ray's accounting.
    Int cpu = 2

    command <<<
        set -euo pipefail

        echo "~{greeting} (shard ~{seed})"
        echo "hostname=$(uname -n)"
        echo "nproc=$(nproc)"

        # stderr on purpose: the driver tails it, so this proves live log streaming.
        echo "shard ~{seed} starting" >&2

        mkdir -p chunks
        for i in 1 2 3; do
            echo "shard=~{seed} chunk=$i" > "chunks/chunk_$i.txt"
        done

        {
            printf 'shard\t%s\n' "~{seed}"
            printf 'host\t%s\n' "$(uname -n)"
            printf 'cpus\t%s\n' "$(nproc)"
        } > report.tsv

        cat chunks/*.txt | cksum | awk '{print $1}' > checksum.txt
    >>>

    output {
        File report = "report.tsv"
        Array[File] chunks = glob("chunks/chunk_*.txt")
        String checksum = read_string("checksum.txt")
    }

    runtime {
        cpu: cpu
        memory: "512 MiB"
        disks: "local-disk 10 HDD"
        maxRetries: 1
        docker: docker
    }
}

task Gather {
    input {
        Array[File] shard_reports
        Array[String] checksums
        String docker
    }

    command <<<
        set -euo pipefail

        cat ~{sep=' ' shard_reports} > summary.tsv
        printf '%s\n' ~{sep=' ' checksums} >> summary.tsv
        wc -l < summary.tsv | tr -d ' ' > total_lines.txt
    >>>

    output {
        File summary = "summary.tsv"
        Int total_lines = read_int("total_lines.txt")
    }

    runtime {
        cpu: 1
        memory: "512 MiB"
        docker: docker
    }
}

# Recipe: "Watch this job and ping me when it's done" ✅

**The ask:** *"I just submitted `<job>`. Watch it, don't spam me, and when it finishes (or dies)
show me the numbers I care about — or the traceback."*

**What Claude does:** attaches a monitor that polls the job's state, **echoes only when the state
changes**, and on any terminal state dumps a **grep-filtered tail of the logs** with exactly the
metrics or failure signatures that matter. You don't babysit; you get woken with the answer. This
is the everyday automation I reach for most — it works for any long job, not just training.

**Skills/tools:** the `Monitor` tool wrapping the `anyscale` CLI. For a truly overnight,
unattended watch, make the Monitor **persistent** so its completion notification re-invokes Claude
(which can then submit the *next* job in a chain, resubmit on failure, etc.).

---

## Run it

The pattern — poll, print on change, surface signal on terminal, match **every** terminal state
(silence is not success):

```bash
jid=prodjob_XXXX
prev=""
while true; do
  s=$(anyscale job status --id "$jid" 2>/dev/null | awk '/^state:|^status:/{print $2}' | head -1)
  [ "$s" != "$prev" ] && echo "[watch] state: $s" && prev=$s
  case "$s" in
    SUCCESS|SUCCEEDED|FAILED|*ERRORED*|TERMINATED|OUT_OF_RETRIES)
      echo "[watch] terminal -> surfacing signal:"
      anyscale job logs --id "$jid" 2>/dev/null \
        | grep -iE "AP |ROC|HR@|metric|P\(|CI|error|traceback|out of memory|unschedulable|capacity" \
        | tail -30
      break;;
  esac
  sleep 90    # scale to job length; the grep filter is the whole trick — tune it per job
done
```

The grep filter is what turns "here's 4,000 lines of log" into "here's your answer." Tune it to the
lines your job actually prints (your metric names) plus the usual failure signatures.

## Real output (I ran this against a finished job)

Pointed at `fintech-fm-paired-bootstrap` (already `SUCCEEDED`), it exited on the first poll and
surfaced the actual result:

```
[watch] state: SUCCEEDED
[watch] terminal -> surfacing metrics + any failures:
[paired] 2,442,779 shared test rows, 2724 frauds, 500 draws, column=embed_xgb
[paired] -> /mnt/user_storage/transaction-fm-v2/downstream/paired_bootstrap_embed_xgb.json
```

That's the whole point: one glance, the numbers and where they were written, no log spelunking.

## Turn it into an automation

- **Chain jobs off completion** (no babysitting): make the Monitor `persistent: true` and describe
  the dependency (*"job B submits when job A reaches a terminal state"*). When A finishes, the
  notification wakes Claude, which reads A's metrics and submits B. This is how "run the analysis
  and launch the next run before I wake up" works.
- **Auto-triage on failure:** in the terminal branch, if the state is `FAILED`/`OUT_OF_RETRIES`,
  hand the job id to `/anyscale-platform-fix` (see the failed-job recipe) instead of just printing
  the traceback.
- **Notify a human:** on terminal, post the surfaced lines to Slack (via MCP) or fire a push
  notification, so the answer finds you instead of the other way around.

> Note: this is event-driven off the monitor's completion notification, not a timer. The "wait"
> is the `sleep` *inside* the loop. Use a timed wake-up only when you're polling something the
> harness can't notify you about (an external CI run, a remote queue).

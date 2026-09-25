package ai.anyscale.nfray

import java.nio.file.FileSystems
import java.nio.file.FileVisitOption
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.PosixFilePermission
import java.util.regex.Pattern
import java.util.stream.Stream

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import nextflow.executor.AbstractGridExecutor
import nextflow.executor.res.AcceleratorResource
import nextflow.processor.TaskRun
import nextflow.util.Duration
import nextflow.util.MemoryUnit
import nextflow.util.ServiceName
import org.pf4j.ExtensionPoint

/**
 * Runs Nextflow processes as Ray tasks.
 *
 * <p>Nextflow already knows how to talk to a batch scheduler: write directives
 * into the job script, submit it with a command, poll a queue, cancel by id. This
 * executor supplies those four things for Ray, so `executor 'ray'` behaves exactly
 * like `executor 'slurm'` from the pipeline's point of view -- which is the whole
 * point. An unmodified nf-core pipeline needs no edits, only a profile.
 *
 * <p>The Ray side lives in Python ({@code nf-ray}, in this template's {@code nf_ray}
 * package) and is reached by running it. That is not a workaround for a missing
 * Java API -- Ray does ship one -- but a deliberate choice. Ray's Java API is
 * documented as "experimental and only supported by the community", its version
 * must match Ray Python exactly, and {@code ray-runtime} loads a JNI library that
 * would have to initialise inside Nextflow's pf4j plugin classloader, in
 * Nextflow's own JVM. Against that, one {@code fork}/{@code exec} per submit buys
 * a plugin small enough to read in one sitting and a Ray integration that survives
 * a Ray upgrade.
 *
 * <p><b>Contract with the Python side.</b> Three things must agree, and each is
 * asserted by the offline unit tests in
 * {@code tests/nextflow-genomics-on-ray/test_nf_ray.py}:
 * <ul>
 *   <li>the {@code #RAY} header format written by {@link #getDirectives}</li>
 *   <li>the {@code Submitted ray task N} line matched by {@link #parseJobId}</li>
 *   <li>the {@code <id> <state>} lines parsed by {@link #parseQueueStatus}</li>
 * </ul>
 */
@Slf4j
@CompileStatic
@ServiceName('ray')
class RayExecutor extends AbstractGridExecutor implements ExtensionPoint {

    private static final Pattern SUBMIT_REGEX = ~/Submitted ray task (\S+)/

    /** Default CLI. Overridable with `ray.cliPath` for a non-standard install. */
    private static final String DEFAULT_CLI = 'nf-ray'

    // -- the scheduler contract ------------------------------------------

    @Override
    protected String getHeaderToken() { '#RAY' }

    /**
     * Emit the resource request as {@code #RAY} header lines.
     *
     * <p><b>The returned list must have an even number of elements.</b>
     * {@code AbstractGridExecutor.getHeaders} walks it as
     * {@code for(i=0; i < size-1; i+=2)}, pairing {@code dir[i]} with
     * {@code dir[i+1]} -- so a stray unpaired element does not raise, it silently
     * drops the last directive. Every {@code <<} below therefore adds exactly two.
     */
    @Override
    protected List<String> getDirectives(TaskRun task, List<String> result) {
        result << '-name' << getJobNameFor(task)
        result << '-cpus' << String.valueOf(task.config.getCpus())

        final MemoryUnit mem = task.config.getMemory()
        if( mem )
            result << '-memory' << String.valueOf(mem.toMega())

        final AcceleratorResource acc = task.config.getAccelerator()
        if( acc ) {
            // `accelerator 2, type: 'nvidia-l4'` -> request=2. AcceleratorResource
            // back-fills request from limit when only one is given, but a process
            // that plainly wants a GPU must never end up asking Ray for zero.
            Integer count = acc.getRequest()
            if( count == null )
                count = acc.getLimit()
            if( count == null )
                count = 1
            result << '-gpus' << String.valueOf(count)
            if( acc.getType() )
                result << '-accelerator' << String.valueOf(acc.getType())
        }

        final Duration time = task.config.getTime()
        if( time )
            result << '-time' << String.valueOf(time.toMinutes())

        // Ray-specific requests Nextflow has no vocabulary for. `ext` is the
        // documented channel for executor-specific settings -- Nextflow does not
        // let a plugin define new process directives.
        //
        // Read with get('ext') rather than getExt(): TaskConfig is a LazyMap and
        // exposes `ext` as a map key, so under @CompileStatic the property form
        // does not resolve to anything.
        final Object extRaw = task.config.get('ext')
        if( extRaw instanceof Map ) {
            final Map ext = (Map) extRaw
            if( ext.get('ray_resources') )
                result << '-resources' << compact(String.valueOf(ext.get('ray_resources')))
            if( ext.get('image') )
                result << '-image' << String.valueOf(ext.get('image'))
        }

        if( result.size() % 2 != 0 )
            throw new IllegalStateException(
                "nf-ray built an odd number of directives (${result.size()}); the last one " +
                "would be silently dropped by AbstractGridExecutor.getHeaders")

        return result
    }

    /**
     * Strip whitespace from a JSON value so it survives header pairing.
     *
     * <p>Each directive occupies one header line, and the Python parser takes the
     * whole remainder of the line as the value -- so a space would in fact
     * survive. This is belt and braces for the case where a future
     * {@code wrapHeader} implementation splits or quotes on whitespace.
     */
    private static String compact(String json) {
        return json.replaceAll(/\s+/, '')
    }

    @Override
    List<String> getSubmitCommandLine(TaskRun task, Path scriptFile) {
        // GridTaskHandler runs this with the task's work directory as CWD, which
        // is why the bare file name is enough -- the same reason `sbatch
        // .command.run` needs no path.
        return cli() + ['submit', scriptFile.getName()]
    }

    // Untyped return, matching the abstract declaration (`abstract parseJobId(String)`)
    // and SlurmExecutor's own override. A narrowed String return is legal Groovy
    // but needlessly differs from every other executor in the tree.
    @Override
    def parseJobId(String text) {
        for( String line : text.readLines() ) {
            final m = SUBMIT_REGEX.matcher(line)
            if( m.find() )
                return m.group(1)
        }
        throw new IllegalArgumentException(
            "Invalid response from `nf-ray submit`:\n$text\n\n" +
            "Expected a line matching /${SUBMIT_REGEX.pattern()}/. If the text above is a\n" +
            "Python traceback, run `nf-ray doctor` -- the usual causes are that Ray is not\n" +
            "reachable or that workDir is not on shared storage.")
    }

    @Override
    protected List<String> getKillCommand() { cli() + ['kill'] }

    @Override
    protected List<String> queueStatusCommand(Object queue) { cli() + ['status'] }

    /**
     * Parse `nf-ray status` output: one {@code <id> <state>} per line.
     *
     * <p>CANCELLED maps to ERROR rather than to a state of its own. Nextflow has
     * no "cancelled" queue status, and the distinction is preserved where it
     * matters anyway: the daemon writes exit code 143 into {@code .exitcode}, so
     * the pipeline's {@code errorStrategy} sees a terminated task rather than a
     * failed one.
     */
    @Override
    protected Map<String, QueueStatus> parseQueueStatus(String text) {
        final Map<String, QueueStatus> result = new LinkedHashMap<String, QueueStatus>()
        if( !text )
            return result
        for( String line : text.readLines() ) {
            final cols = line.trim().split(/\s+/)
            if( cols.length < 2 )
                continue
            result.put(cols[0], decode(cols[1]))
        }
        return result
    }

    private static QueueStatus decode(String state) {
        switch( state ) {
            case 'PENDING':   return QueueStatus.PENDING
            case 'RUNNING':   return QueueStatus.RUNNING
            case 'DONE':      return QueueStatus.DONE
            case 'ERROR':     return QueueStatus.ERROR
            case 'CANCELLED': return QueueStatus.ERROR
            default:          return QueueStatus.UNKNOWN
        }
    }

    // -- the pipeline's bin/ ---------------------------------------------

    /** The copy of the project's bin/ that tasks run; null when there is none to use. */
    private Path stagedBinDir

    /**
     * Where tasks find the pipeline's bin/ scripts.
     *
     * <p>Nextflow appends this to every task's PATH (TaskProcessor.getProcessEnvironment),
     * and the default is {@code session.binDir}, i.e. {@code <projectDir>/bin}, a path on
     * the machine running Nextflow. A grid executor gets away with that on HPC, where the
     * project sits on a filesystem every node mounts. A Ray cluster shares only
     * /mnt/cluster_storage: the pipeline lives in the workspace's or the job's working
     * directory on the head node, so the first process to call a bin/ script failed on a
     * worker with `collect_vcfeval.py: command not found`, exit 127. Tasks get a copy under
     * the work directory instead, which every node must see anyway. The AWS Batch executor
     * does the same with S3.
     *
     * <p>Module-level bin directories ({@code nextflow.enable.moduleBinaries}) are not
     * copied.
     */
    @Override
    Path getBinDir() {
        return stagedBinDir ?: super.getBinDir()
    }

    /**
     * Copy {@code session.binDir} to a fresh {@code <workDir>/tmp/..../bin}.
     *
     * <p>Every file in the copy is made executable. rayapp's template zip records no Unix
     * modes, so a template unpacked from it (CI, or a console template) has its bin/
     * scripts at 0644, and bash refuses a non-executable script it finds on PATH with
     * `Permission denied`, exit 126. Symlinks are followed, since a link back into the
     * project would dangle on a worker.
     *
     * <p>Returns null, leaving tasks on the original path, when the pipeline has no bin/,
     * when {@code executor.disableRemoteBinDir} says the project is already on shared
     * storage, or when the work directory is not a local POSIX path.
     */
    private Path stageBinDir() {
        final Path source = session?.getBinDir()
        if( source == null || !Files.isDirectory(source) )
            return null
        if( session.disableRemoteBinDir ) {
            log.debug "nf-ray: executor.disableRemoteBinDir is set; tasks use ${source}"
            return null
        }
        if( getWorkDir().getFileSystem() != FileSystems.getDefault() )
            return null
        try {
            final Path target = getTempDir('bin')
            copyExecutable(source, target)
            log.info "nf-ray: copied ${source} to ${target}, where every node can run it"
            return target
        }
        catch( IOException | UncheckedIOException e ) {
            log.warn "nf-ray: could not copy ${source} to the work directory (${e.message}). " +
                     "Tasks will look for bin/ scripts in ${source}, which only this node has."
            return null
        }
    }

    private static void copyExecutable(Path source, Path target) throws IOException {
        final Set<PosixFilePermission> exec = EnumSet.of(
            PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_EXECUTE,
            PosixFilePermission.GROUP_READ, PosixFilePermission.GROUP_EXECUTE,
            PosixFilePermission.OTHERS_READ, PosixFilePermission.OTHERS_EXECUTE)
        final Stream<Path> walk = Files.walk(source, FileVisitOption.FOLLOW_LINKS)
        try {
            final Iterator<Path> it = walk.iterator()
            while( it.hasNext() ) {
                final Path path = it.next()
                final Path dest = target.resolve(source.relativize(path).toString())
                if( Files.isDirectory(path) ) {
                    Files.createDirectories(dest)
                    continue
                }
                Files.copy(path, dest, StandardCopyOption.REPLACE_EXISTING)
                try {
                    final Set<PosixFilePermission> perms = EnumSet.noneOf(PosixFilePermission)
                    perms.addAll(Files.getPosixFilePermissions(dest))
                    perms.addAll(exec)
                    Files.setPosixFilePermissions(dest, perms)
                }
                catch( IOException | UnsupportedOperationException e ) {
                    // Still worth using: the copy was created with the source's mode.
                    log.debug "nf-ray: could not chmod ${dest}: ${e.message}"
                }
            }
        }
        finally {
            walk.close()
        }
    }

    // -- lifecycle -------------------------------------------------------

    @Override
    protected void register() {
        super.register()
        // Before any task exists: TaskProcessor reads getBinDir() once per process and
        // keeps the answer.
        stagedBinDir = stageBinDir()
        // Start the daemon now rather than on the first submit. Connecting to Ray
        // is the step most likely to fail, and failing here costs seconds at
        // pipeline start instead of surfacing as an unexplained submit error once
        // the first process has already staged its inputs.
        final cmd = cli() + ['up']
        try {
            final proc = new ProcessBuilder(cmd).redirectErrorStream(true).start()
            final out = proc.inputStream.text
            if( proc.waitFor() != 0 )
                log.warn "nf-ray: could not start the Ray daemon at registration:\n${out}\n" +
                         "The first task submission will try again."
            else
                log.debug "nf-ray: ${out.trim()}"
        }
        catch( Exception e ) {
            log.warn "nf-ray: could not run `${DEFAULT_CLI}` (${e.message}). " +
                     "Is it on PATH? Try `nf-ray doctor`."
        }
    }

    @Override
    void shutdown() {
        // Release the Ray driver promptly. The daemon also has an idle timeout, but
        // that is a backstop for a killed run -- on a clean exit the cluster should
        // be free to scale down as soon as the pipeline is done with it.
        try {
            new ProcessBuilder(cli() + ['down']).redirectErrorStream(true).start().waitFor()
        }
        catch( Exception e ) {
            log.debug "nf-ray: shutdown call failed: ${e.message}"
        }
        super.shutdown()
    }

    // -- configuration ---------------------------------------------------

    /**
     * The CLI invocation, with the `ray` config scope forwarded as environment.
     *
     * <p>Every setting has to reach three separate processes -- this JVM, the
     * daemon, and a fresh CLI per submit -- and only the environment gets to all
     * three. Java cannot mutate its own environment, so the settings are pushed
     * through {@code env} on each call instead, which has the side benefit that
     * the full configuration is visible in {@code .command.log} when a submit
     * fails.
     */
    private List<String> cli() {
        final List<String> argv = ['/usr/bin/env']
        rayEnv().each { String k, String v -> argv << "${k}=${v}".toString() }
        argv << cliPath()
        return argv
    }

    private String cliPath() {
        final Object configured = rayScope().get('cliPath')
        return configured ? String.valueOf(configured) : DEFAULT_CLI
    }

    private Map<String, Object> rayScope() {
        final Object scope = session?.config?.get('ray')
        return (scope instanceof Map) ? (Map<String, Object>) scope : [:] as Map<String, Object>
    }

    /**
     * Translate the `ray { }` config scope into `NF_RAY_*` variables.
     *
     * <p>{@code clampResources} becomes {@code NF_RAY_CLAMP_RESOURCES}, and so on,
     * so the scope and the environment are two spellings of one setting rather
     * than two lists to keep in step.
     */
    private Map<String, String> rayEnv() {
        final Map<String, String> out = new LinkedHashMap<String, String>()
        rayScope().each { Object k, Object v ->
            final String key = String.valueOf(k)
            if( key == 'cliPath' )
                return
            out.put('NF_RAY_' + camelToUpper(key), String.valueOf(v))
        }
        // Always last, so it cannot be shadowed by the scope: the daemon keys its
        // socket on this path, and two runs pointing at one socket would share a
        // Ray driver and each other's task ids.
        if( session?.workDir != null )
            out.put('NF_RAY_WORK_DIR', session.workDir.toString())
        return out
    }

    private static String camelToUpper(String name) {
        return name.replaceAll(/([a-z0-9])([A-Z])/, '$1_$2').toUpperCase()
    }
}

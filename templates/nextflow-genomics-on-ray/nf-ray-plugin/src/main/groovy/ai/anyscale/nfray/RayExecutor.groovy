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

/** Grid executor that runs each Nextflow task as a Ray task through the {@code nf-ray} CLI. */
@Slf4j
@CompileStatic
@ServiceName('ray')
class RayExecutor extends AbstractGridExecutor implements ExtensionPoint {

    // The #RAY header, this line and the status lines are a contract with nf_ray (test_nf_ray.py).
    private static final Pattern SUBMIT_REGEX = ~/Submitted ray task (\S+)/

    private static final String DEFAULT_CLI = 'nf-ray'

    @Override
    protected String getHeaderToken() { '#RAY' }

    // getHeaders pairs the list's elements, so every << adds exactly two.
    @Override
    protected List<String> getDirectives(TaskRun task, List<String> result) {
        result << '-name' << getJobNameFor(task)
        result << '-cpus' << String.valueOf(task.config.getCpus())

        final MemoryUnit mem = task.config.getMemory()
        if( mem )
            result << '-memory' << String.valueOf(mem.toMega())

        final AcceleratorResource acc = task.config.getAccelerator()
        if( acc ) {
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

        // get('ext'), not getExt(): TaskConfig is a LazyMap, and under @CompileStatic the
        // property form resolves to nothing.
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

    private static String compact(String json) {
        return json.replaceAll(/\s+/, '')
    }

    @Override
    List<String> getSubmitCommandLine(TaskRun task, Path scriptFile) {
        // Run from the task's work directory, as `sbatch .command.run` is.
        return cli() + ['submit', scriptFile.getName()]
    }

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

    /** Parse {@code nf-ray status}: one {@code <id> <state>} per line. */
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
            // Nextflow has no cancelled state; the daemon's .exitcode 143 tells them apart.
            case 'CANCELLED': return QueueStatus.ERROR
            default:          return QueueStatus.UNKNOWN
        }
    }

    private Path stagedBinDir

    /** A copy of bin/ in the shared work directory: the project exists only on the head node. */
    @Override
    Path getBinDir() {
        return stagedBinDir ?: super.getBinDir()
    }

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

    // chmod +x: rayapp's zip records no modes, so bin/ can arrive 0644. Links are followed, since
    // one back into the project would dangle on a worker.
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
                    log.debug "nf-ray: could not chmod ${dest}: ${e.message}"
                }
            }
        }
        finally {
            walk.close()
        }
    }

    @Override
    protected void register() {
        super.register()
        // Before any task: TaskProcessor reads getBinDir() once per process.
        stagedBinDir = stageBinDir()
        // Connecting to Ray is what most often fails, so do it at start, not on the first submit.
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
        // Release the driver now; the daemon's idle timeout is a backstop for killed runs.
        try {
            new ProcessBuilder(cli() + ['down']).redirectErrorStream(true).start().waitFor()
        }
        catch( Exception e ) {
            log.debug "nf-ray: shutdown call failed: ${e.message}"
        }
        super.shutdown()
    }

    // The `ray` scope goes through `env` on every call: only the environment reaches the daemon
    // and each CLI process, and Java cannot set its own.
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

    private Map<String, String> rayEnv() {
        final Map<String, String> out = new LinkedHashMap<String, String>()
        rayScope().each { Object k, Object v ->
            final String key = String.valueOf(k)
            if( key == 'cliPath' )
                return
            out.put('NF_RAY_' + camelToUpper(key), String.valueOf(v))
        }
        // Last, so the scope cannot shadow it: the daemon keys its socket on this path.
        if( session?.workDir != null )
            out.put('NF_RAY_WORK_DIR', session.workDir.toString())
        return out
    }

    private static String camelToUpper(String name) {
        return name.replaceAll(/([a-z0-9])([A-Z])/, '$1_$2').toUpperCase()
    }
}

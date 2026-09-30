package ai.anyscale.nfray

import groovy.transform.CompileStatic
import nextflow.plugin.BasePlugin
import org.pf4j.PluginWrapper

/** Plugin entry point, required by pf4j; the executor is {@link RayExecutor}. */
@CompileStatic
class RayPlugin extends BasePlugin {

    RayPlugin(PluginWrapper wrapper) {
        super(wrapper)
    }
}

package ai.anyscale.nfray

import groovy.transform.CompileStatic
import nextflow.plugin.BasePlugin
import org.pf4j.PluginWrapper

/**
 * Entry point for the `nf-ray` plugin.
 *
 * Nothing to do beyond existing: the plugin contributes one extension point,
 * {@link RayExecutor}, which Nextflow finds through the `extensionPoints` list in
 * build.gradle and selects by the name on its {@code @ServiceName} annotation.
 * The class is still required -- `className` in build.gradle is mandatory and
 * pf4j instantiates it to manage the plugin's lifecycle.
 */
@CompileStatic
class RayPlugin extends BasePlugin {

    RayPlugin(PluginWrapper wrapper) {
        super(wrapper)
    }
}

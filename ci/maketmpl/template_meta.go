package maketmpl

type templateMeta struct {
	Name string `json:"name"`

	// Base64 encoded cluster environment.
	ClusterEnvBase64 string `json:"cluster_env_base64,omitempty"`

	// Base64 encoded compute config for different cloud types.
	ComputeConfigBase64 map[string]string `json:"compute_config_base64,omitempty"`
}

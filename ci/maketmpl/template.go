package maketmpl

// Template defines the definition of a workspace template.
type Template struct {
	Name string `yaml:"name"`
	Dir  string `yaml:"dir"`

	Emoji       string `yaml:"emoji"`
	Title       string `yaml:"title"`
	Description string `yaml:"description"`

	ClusterEnv map[string]any `yaml:"cluster_env"`

	// A map of files for different compute platforms.
	ComputeConfig map[string]string `yaml:"compute_config"`
}

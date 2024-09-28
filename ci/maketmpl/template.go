package maketmpl

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

// Template defines the definition of a workspace template.
type Template struct {
	Name string `yaml:"name"`
	Dir  string `yaml:"dir"`

	Emoji       string `yaml:"emoji,omitempty"`
	Title       string `yaml:"title,omitempty"`
	Description string `yaml:"description,omitempty"`

	ClusterEnv map[string]any `yaml:"cluster_env"`

	// A map of files for different compute platforms.
	ComputeConfig map[string]string `yaml:"compute_config"`
}

func readTemplates(yamlFile string) ([]*Template, error) {
	var tmpls []*Template

	bs, err := os.ReadFile(yamlFile)
	if err != nil {
		return nil, fmt.Errorf("read file %q: %w", yamlFile, err)
	}
	if err := yaml.Unmarshal(bs, &tmpls); err != nil {
		return nil, fmt.Errorf("unmarshal yaml: %w", err)
	}
	return tmpls, nil
}

package maketmpl

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v2"
)

// ClusterEnvBYOD is the cluster environment for BYOD clusters.
type ClusterEnvBYOD struct {
	DockerImage string `yaml:"docker_image" json:"docker_image"`
	RayVersion  string `yaml:"ray_version,omitempty" json:"ray_version,omitempty"`
}

// ClusterEnv is the cluster environment for Anyscale clusters.
type ClusterEnv struct {
	BuildID string `yaml:"build_id,omitempty" json:"build_id,omitempty"`

	// BYOD is the cluster environment for bring-your-own-docker clusters.
	BYOD *ClusterEnvBYOD `yaml:"byod,omitempty" json:"byod,omitempty"`
}

// Template defines the definition of a workspace template.
type Template struct {
	Name string `yaml:"name" json:"name"`
	Dir  string `yaml:"dir" json:"dir"`

	Emoji       string `yaml:"emoji" json:"emoji"`
	Title       string `yaml:"title" json:"title"`
	Description string `yaml:"description,omitempty" json:"description,omitempty"`

	ClusterEnv *ClusterEnv `yaml:"cluster_env" json:"cluster_env"`

	// A map of files for different compute platforms.
	ComputeConfig map[string]string `yaml:"compute_config" json:"compute_config"`
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

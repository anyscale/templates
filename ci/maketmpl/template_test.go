package maketmpl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const testBuildDotYaml = `
# Just for testing

- name: job-intro
  emoji: 🔰
  title: Intro to Jobs
  description: Introduction on how to use Anyscale Jobs
  dir: templates/intro-jobs
  cluster_env:
    build_id: anyscaleray2340-py311
  compute_config:
    GCP: configs/basic-single-node/gce.yaml
    AWS: configs/basic-single-node/aws.yaml

- name: workspace-intro
  emoji: 🔰
  title: Intro to Workspaces
  description: Introduction on how to use Anyscale Workspaces
  dir: templates/intro-workspaces
  cluster_env:
    byod:
      docker_image: cr.ray.io/ray:2340-py311
      ray_version: 2.34.0
  compute_config:
    GCP: configs/basic-single-node/gce.yaml
    AWS: configs/basic-single-node/aws.yaml
`

func TestReadTemplates(t *testing.T) {
	tmp := t.TempDir()

	f := filepath.Join(tmp, "BUILD.yaml")
	if err := os.WriteFile(f, []byte(testBuildDotYaml), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	got, err := readTemplates(f)
	if err != nil {
		t.Fatalf("readTemplates(%q): %v", f, err)
	}

	want := []*Template{{
		Name:        "job-intro",
		Emoji:       "🔰",
		Title:       "Intro to Jobs",
		Dir:         "templates/intro-jobs",
		Description: "Introduction on how to use Anyscale Jobs",
		ClusterEnv:  &ClusterEnv{BuildID: "anyscaleray2340-py311"},
		ComputeConfig: map[string]string{
			"GCP": "configs/basic-single-node/gce.yaml",
			"AWS": "configs/basic-single-node/aws.yaml",
		},
	}, {
		Name:        "workspace-intro",
		Emoji:       "🔰",
		Title:       "Intro to Workspaces",
		Dir:         "templates/intro-workspaces",
		Description: "Introduction on how to use Anyscale Workspaces",
		ClusterEnv: &ClusterEnv{
			BYOD: &ClusterEnvBYOD{
				DockerImage: "cr.ray.io/ray:2340-py311",
				RayVersion:  "2.34.0",
			},
		},
		ComputeConfig: map[string]string{
			"GCP": "configs/basic-single-node/gce.yaml",
			"AWS": "configs/basic-single-node/aws.yaml",
		},
	}}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("readTemplates(%q), got %+v, want %+v", f, got, want)
	}

	// Loopback with JSON encoding, and it should be the same.
	jsonBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal templates: %v", err)
	}

	jsonFile := filepath.Join(tmp, "BUILD.json")
	if err := os.WriteFile(jsonFile, jsonBytes, 0o600); err != nil {
		t.Fatalf("write json file: %v", err)
	}

	jsonGot, err := readTemplates(jsonFile)
	if err != nil {
		t.Fatalf("read json file: %v", err)
	}
	if !reflect.DeepEqual(jsonGot, want) {
		t.Errorf("read json file, got %+v, want %+v", jsonGot, want)
	}
}

func TestReadTemplates_withError(t *testing.T) {
	tmp := t.TempDir()
	f := filepath.Join(tmp, "BUILD.yaml")
	if _, err := readTemplates(f); err == nil {
		t.Error("want error, got nil")
	}
}

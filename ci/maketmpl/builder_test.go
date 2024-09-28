package maketmpl

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"testing"
)

func readFileInZip(f *zip.File) ([]byte, error) {
	rc, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, rc); err != nil {
		return nil, err
	}
	if err := rc.Close(); err != nil {
		return nil, fmt.Errorf("close: %w", err)
	}
	return buf.Bytes(), nil
}

func fileNameList(files map[string][]byte) []string {
	var names []string
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func TestBuilder(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping on windows")
	}

	checkJupyterOrSkipOnLocal(t)

	tmp := t.TempDir()

	tmpl := &Template{
		Name:  "reefy-ray",
		Dir:   "reefy-ray",
		Title: "Ray lives in the REEf",
		ClusterEnv: map[string]any{
			"build_id": "anyscaleray2340-py311",
		},
		ComputeConfig: map[string]string{
			"AWS": "configs/aws.yaml",
			"GCP": "configs/gcp.yaml",
		},
	}

	// Build the template, save in tmp dir.
	b := newBuilder(tmpl, "testdata")
	if err := b.build(tmp); err != nil {
		t.Fatal("build: ", err)
	}

	// Check the output files.
	buildZip := filepath.Join(tmp, buildDotZip)

	z, err := zip.OpenReader(buildZip)
	if err != nil {
		t.Fatalf("open built zip: %v", err)
	}
	defer z.Close()

	gotFiles := make(map[string][]byte)

	for _, f := range z.File {
		content, err := readFileInZip(f)
		if err != nil {
			t.Fatalf("read file %q: %v", f.Name, err)
		}
		gotFiles[f.Name] = content
	}

	gotFileNames := fileNameList(gotFiles)

	srcNotebook, err := os.ReadFile("testdata/reefy-ray/README.ipynb")
	if err != nil {
		t.Fatal("read src readme: ", err)
	}
	srcReadme, err := os.ReadFile("testdata/reefy-ray/README.md")
	if err != nil {
		t.Fatal("read src readme: ", err)
	}
	srcPng, err := os.ReadFile("testdata/reefy-ray/a.png")
	if err != nil {
		t.Fatal("read src png: ", err)
	}

	wantFiles := map[string][]byte{
		"README.ipynb":       srcNotebook,
		"README.md":          srcReadme,
		"a.png":              srcPng,
		".meta/ray-app.json": nil,
	}
	wantFileNames := fileNameList(wantFiles)

	if !reflect.DeepEqual(gotFileNames, wantFileNames) {
		t.Fatalf("got files %v, want %v", gotFileNames, wantFileNames)
	}

	for name, want := range wantFiles {
		if want == nil {
			continue
		}
		got := gotFiles[name]
		if !bytes.Equal(got, want) {
			t.Errorf("file %q: got %q, want %q", name, got, want)
		}
	}

	// Check the meta file in the zip.
	metaGot := new(templateMeta)
	metaGotBytes := gotFiles[".meta/ray-app.json"]
	if err := json.Unmarshal(metaGotBytes, metaGot); err != nil {
		t.Fatalf("unmarshal meta json: %v", err)
	}

	if metaGot.Name != "reefy-ray" {
		t.Errorf("meta name: got %q, want %q", metaGot.Name, "reefy-ray")
	}

	clusterEnv, err := base64.StdEncoding.DecodeString(
		metaGot.ClusterEnvBase64,
	)
	if err != nil {
		t.Fatalf("decode cluster env: %v", err)
	}

	var clusterEnvGot map[string]any
	if err := json.Unmarshal(clusterEnv, &clusterEnvGot); err != nil {
		t.Fatalf("unmarshal cluster env: %v", err)
	}
	if !reflect.DeepEqual(clusterEnvGot, tmpl.ClusterEnv) {
		t.Errorf(
			"cluster env: got %+v, want %+v",
			clusterEnvGot, tmpl.ClusterEnv,
		)
	}

	// Check the content of the compute configs.
	gcpComputeConfig, err := os.ReadFile("testdata/configs/gcp.yaml")
	if err != nil {
		t.Fatal("read gcp compute config: ", err)
	}
	awsComputeConfig, err := os.ReadFile("testdata/configs/aws.yaml")
	if err != nil {
		t.Fatal("read aws compute config: ", err)
	}

	gcpComputeConfigGot, err := base64.StdEncoding.DecodeString(
		metaGot.ComputeConfigBase64["GCP"],
	)
	if err != nil {
		t.Fatalf("decode gcp compute config: %v", err)
	}
	if !bytes.Equal(gcpComputeConfigGot, gcpComputeConfig) {
		t.Errorf(
			"gcp compute config: got %q, want %q",
			gcpComputeConfigGot, gcpComputeConfig,
		)
	}

	awsComputeConfigGot, err := base64.StdEncoding.DecodeString(
		metaGot.ComputeConfigBase64["AWS"],
	)
	if err != nil {
		t.Fatalf("decode aws compute config: %v", err)
	}
	if !bytes.Equal(awsComputeConfigGot, awsComputeConfig) {
		t.Errorf(
			"aws compute config: got %q, want %q",
			awsComputeConfigGot, awsComputeConfig,
		)
	}

	// Check the external app meta file is the same as the one in the zip.
	externalAppMeta, err := os.ReadFile(filepath.Join(tmp, rayAppDotJSON))
	if err != nil {
		t.Fatalf("read external app meta: %v", err)
	}
	if !bytes.Equal(metaGotBytes, externalAppMeta) {
		t.Errorf(
			"external app meta: got %q, want %q",
			metaGotBytes, externalAppMeta,
		)
	}

	// Check the generated readme files.
	// Only perform checks on some critical properties.

	// First check the readme generated for docs.
	// This is generated from the markdown file.
	{
		doc, err := os.ReadFile(filepath.Join(tmp, readmeDocMD))
		if err != nil {
			t.Fatalf("read generated readme: %v", err)
		}
		if !bytes.Contains(doc, []byte(`print("this is just an example")`)) {
			t.Errorf("readme for doc %q, missing python code", doc)
		}
		if !bytes.Contains(doc, []byte("extra manual line")) {
			t.Errorf("readme for doc %q, missing the manual line", doc)
		}
		if bytes.Contains(doc, []byte("a.png")) {
			t.Errorf("readme for doc %q, png file not inlined", doc)
		}
	}

	// Next check the readme converted from the notebook.
	{
		nb, err := os.ReadFile(filepath.Join(tmp, readmeNotebookGitHubMD))
		if err != nil {
			t.Fatalf("read generated readme: %v", err)
		}

		// We are emulating the situation where a user generates the README
		// from the notebook, and then manually edits it (by appending an
		// extra line), and saved it in the template's source directory.
		// As a result, the generated README from the notebook should be a
		// prefix of the README from the source directory.
		if !bytes.HasPrefix(srcReadme, nb) {
			t.Errorf(
				"readme generated from notebook %q, is not the prefix of %q",
				nb, srcReadme,
			)
		}
		if !bytes.Contains(nb, []byte(`print("this is just an example")`)) {
			t.Errorf("readme for doc %q, missing python code", nb)
		}
	}

	// Finally, check the readme cleaned up for GitHub.
	// This should be exactly the same as the saved markdown file
	// from the input.
	{
		gh, err := os.ReadFile(filepath.Join(tmp, readmeGitHubMD))
		if err != nil {
			t.Fatalf("read generated readme: %v", err)
		}

		if !bytes.Equal(gh, srcReadme) {
			t.Errorf(
				"readme for github %q, does not match the saved %q",
				gh, srcReadme,
			)
		}
	}
}
